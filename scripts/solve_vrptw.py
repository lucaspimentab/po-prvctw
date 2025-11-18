"""
Solve Solomon VRPTW instances with OR-Tools and compare against best-known values.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# Instâncias padrão
INSTANCE_SPECS: List[Tuple[str, Path]] = [
    ("C101", Path("instances") / "Solomon_C101.txt"),
    ("R101", Path("instances") / "Solomon_R101.txt"),
    ("RC101", Path("instances") / "Solomon_RC101.txt"),
]

BKS_TABLE_PATH = Path("data") / "solomon_bks.csv"  # contém veículos/distâncias ótimas publicadas
RESULTS_DIR = Path("results")
DEFAULT_TIME_LIMIT = 120
INSTANCE_TIME_LIMITS = {
    "C101": 60,
    "R101": 240,
    "RC101": 240,
}



@dataclass
class InstanceData:
    """Estrutura com todos os dados da instância."""

    name: str
    capacity: int
    num_vehicles: int
    coordinates: List[Tuple[float, float]]
    demands: List[int]
    time_windows: List[Tuple[int, int]]
    service_times: List[int]



@dataclass
class RouteResult:
    """Resumo de uma rota individual."""

    vehicle_id: int
    nodes: List[int]
    distance: float
    load: int
    start_time: int
    end_time: int



@dataclass
class InstanceResult:
    """Resultados agregados por instância."""

    instance: str
    vehicles_available: int
    vehicles_used: int
    capacity: int
    total_distance: float
    best_known_distance: float
    best_known_vehicles: int
    distance_gap_pct: float
    vehicle_gap: int
    runtime_seconds: float
    routes: List[RouteResult]


def parse_solomon_instance(path: Path) -> InstanceData:
    """Lê um .txt de Solomon e devolve os parâmetros do PRVC-TW."""
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    lines_iter = iter(lines)
    name = next(lines_iter).strip()

    # Avança até a parte de veículos
    for line in lines_iter:
        if line.strip().upper().startswith("VEHICLE"):
            break
    # Skip header
    for line in lines_iter:
        if line.strip().startswith("NUMBER"):
            break
    vehicle_line = next(lines_iter)
    number_str, capacity_str = vehicle_line.split()
    num_vehicles = int(number_str)
    capacity = int(capacity_str)

    for line in lines_iter:
        if line.strip().upper().startswith("CUSTOMER"):
            break

    # Pula o cabeçalho da tabela de clientes
    header = next(lines_iter)
    assert "CUST" in header.upper(), "Unexpected Solomon file header."

    coordinates: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    time_windows: Dict[int, Tuple[int, int]] = {}
    service_times: Dict[int, int] = {}

    for raw in lines_iter:
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 7:
            continue
        node_id = int(parts[0])
        x_coord = float(parts[1])
        y_coord = float(parts[2])
        demand = int(parts[3])
        ready_time = int(parts[4])
        due_time = int(parts[5])
        service_time = int(parts[6])
        coordinates[node_id] = (x_coord, y_coord)
        demands[node_id] = demand
        time_windows[node_id] = (ready_time, due_time)
        service_times[node_id] = service_time

    # Reordena os vetores para que o índice do cliente bata com a posição na lista
    node_ids = sorted(coordinates.keys())
    coordinates_list = [coordinates[i] for i in node_ids]
    demands_list = [demands[i] for i in node_ids]
    time_windows_list = [time_windows[i] for i in node_ids]
    service_list = [service_times[i] for i in node_ids]

    return InstanceData(
        name=name,
        capacity=capacity,
        num_vehicles=num_vehicles,
        coordinates=coordinates_list,
        demands=demands_list,
        time_windows=time_windows_list,
        service_times=service_list,
    )


def load_bks_table(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return {
            row["instance"].strip().upper(): {
                "distance": float(row["best_known_distance"]),
                "vehicles": int(row["best_known_vehicles"]),
                "reference": row["bks_reference"],
            }
            for row in reader
        }


def build_distance_matrices(coords: List[Tuple[float, float]]) -> Tuple[List[List[int]], List[List[float]]]:
    size = len(coords)
    dist_int = [[0] * size for _ in range(size)]
    dist_float = [[0.0] * size for _ in range(size)]
    for i in range(size):
        x_i, y_i = coords[i]
        for j in range(size):
            x_j, y_j = coords[j]
            dist = math.hypot(x_i - x_j, y_i - y_j)
            dist_float[i][j] = dist
            dist_int[i][j] = int(round(dist))
    return dist_int, dist_float


def solve_instance(data: InstanceData, time_limit_sec: int = DEFAULT_TIME_LIMIT) -> InstanceResult:
    """Resolve a instância recebida usando o solver de roteamento do OR-Tools."""
    time_matrix_int, distance_matrix = build_distance_matrices(data.coordinates)
    manager = pywrapcp.RoutingIndexManager(len(time_matrix_int), data.num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit_cache: Dict[Tuple[int, int], int] = {}

    def transit_callback(from_index: int, to_index: int) -> int:
        # Tempo de deslocamento + serviço; usado tanto como custo quanto como dimensão
        key = (from_index, to_index)
        if key in transit_cache:
            return transit_cache[key]
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = time_matrix_int[from_node][to_node]
        service = data.service_times[from_node]
        transit_cache[key] = travel + service
        return transit_cache[key]

    transit_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.SetFixedCostOfAllVehicles(1000)  # peso alto garante prioridade para reduzir nº de veículos

    max_due = max(high for _, high in data.time_windows)
    # Dimensão de tempo -> garante respeito às janelas [a_i, b_i] e propagação de sequência
    routing.AddDimension(
        transit_index,
        max_due,
        max_due,
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for node, (start, end) in enumerate(data.time_windows):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(start, end)

    # Reconstrói cada rota a partir da solução inteira
    for vehicle_id in range(data.num_vehicles):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        depot_window = data.time_windows[0]
        time_dimension.CumulVar(start_index).SetRange(*depot_window)
        time_dimension.CumulVar(end_index).SetRange(*depot_window)

    def demand_callback(from_index: int) -> int:
        # Demanda do cliente associado ao índice
        node = manager.IndexToNode(from_index)
        return data.demands[node]

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    # Dimensão de capacidade -> aplica Σ demandas ≤ Q para cada veículo
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [data.capacity] * data.num_vehicles,
        True,
        "Capacity",
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.time_limit.FromSeconds(time_limit_sec)
    search_params.log_search = False
    # Estratégias padrão (solução inicial barata + busca local guiada)
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    start_time = time.perf_counter()
    solution = routing.SolveWithParameters(search_params)
    runtime = time.perf_counter() - start_time

    if solution is None:
        raise RuntimeError(f"No feasible solution found for {data.name}.")

    routes: List[RouteResult] = []
    total_distance = 0.0
    vehicles_used = 0

    for vehicle_id in range(data.num_vehicles):
        index = routing.Start(vehicle_id)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
        vehicles_used += 1
        route_nodes = [manager.IndexToNode(index)]
        route_distance = 0.0
        route_load = 0
        start_time_val = solution.Value(time_dimension.CumulVar(index))

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_load += data.demands[node]
            next_index = solution.Value(routing.NextVar(index))
            if routing.IsEnd(next_index):
                break
            next_node = manager.IndexToNode(next_index)
            route_nodes.append(next_node)
            route_distance += distance_matrix[node][next_node]
            index = next_index

        last_node = manager.IndexToNode(index)
        route_distance += distance_matrix[last_node][0]

        end_time_val = solution.Value(time_dimension.CumulVar(index))
        total_distance += route_distance
        routes.append(
            RouteResult(
                vehicle_id=vehicle_id,
                nodes=route_nodes + [0],
                distance=round(route_distance, 2),
                load=route_load,
                start_time=start_time_val,
                end_time=end_time_val,
            )
        )

    return InstanceResult(
        instance=data.name,
        vehicles_available=data.num_vehicles,
        vehicles_used=vehicles_used,
        capacity=data.capacity,
        total_distance=round(total_distance, 2),
        best_known_distance=0.0,
        best_known_vehicles=0,
        distance_gap_pct=0.0,
        vehicle_gap=0,
        runtime_seconds=round(runtime, 2),
        routes=routes,
    )


def attach_bks_info(result: InstanceResult, bks_row: Dict[str, float]) -> InstanceResult:
    result.best_known_distance = bks_row["distance"]
    result.best_known_vehicles = int(bks_row["vehicles"])
    result.distance_gap_pct = round(
        ((result.total_distance - result.best_known_distance) / result.best_known_distance) * 100.0,
        2,
    )
    result.vehicle_gap = result.vehicles_used - result.best_known_vehicles
    return result


def save_results(results: Iterable[InstanceResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialisable = []
    for res in results:
        payload = asdict(res)
        payload["routes"] = [asdict(route) for route in res.routes]
        serialisable.append(payload)
    destination.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def export_summary_csv(results: Iterable[InstanceResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance",
        "vehicles_available",
        "vehicles_used",
        "best_known_vehicles",
        "vehicle_gap",
        "total_distance",
        "best_known_distance",
        "distance_gap_pct",
        "runtime_seconds",
    ]
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "instance": res.instance,
                    "vehicles_available": res.vehicles_available,
                    "vehicles_used": res.vehicles_used,
                    "best_known_vehicles": res.best_known_vehicles,
                    "vehicle_gap": res.vehicle_gap,
                    "total_distance": res.total_distance,
                    "best_known_distance": res.best_known_distance,
                    "distance_gap_pct": res.distance_gap_pct,
                    "runtime_seconds": res.runtime_seconds,
                }
            )


def main() -> None:
    bks_table = load_bks_table(BKS_TABLE_PATH)
    collected: List[InstanceResult] = []

    for instance_name, source in INSTANCE_SPECS:
        data = parse_solomon_instance(source)
        print(f"Solving {instance_name} ({source}) ...")
        time_limit = INSTANCE_TIME_LIMITS.get(instance_name.upper(), DEFAULT_TIME_LIMIT)
        result = solve_instance(data, time_limit_sec=time_limit)
        bks_info = bks_table.get(instance_name.upper())
        if bks_info:
            result = attach_bks_info(result, bks_info)
        else:
            print(f"Warning: no BKS information for {instance_name}.")
        print(
            f"  -> used {result.vehicles_used}/{result.vehicles_available} vehicles, "
            f"distance {result.total_distance:.2f} (gap {result.distance_gap_pct:.2f}%)."
        )
        collected.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(collected, RESULTS_DIR / "solomon_vrptw_results.json")
    export_summary_csv(collected, RESULTS_DIR / "solomon_vrptw_results.csv")


if __name__ == "__main__":
    main()
