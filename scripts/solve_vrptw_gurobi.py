"""
Solver do PRVC-TW (Gurobi) nas instancias de Solomon (C101, R101, RC101).

Modelo:
  Variaveis:
    x_ij  -> 1 se o arco (i,j) e percorrido em alguma rota.
    load_i -> carga acumulada ao sair de i (MTZ).
    t_i   -> tempo acumulado ate i (usa janela do deposito como limite global).

  Funcao objetivo:
    min  sum_{i} sum_{j} c_ij * x_ij

  Restricoes principais:
    (1) Atendimento unico do cliente: sum_j x_ij = 1 e sum_j x_ji = 1.
    (2) Saida/retorno no deposito + conservacao do fluxo.
    (3) Capacidade e eliminacao de subtours (restricoes MTZ em load_i).
    (4) Janela de tempo global baseada em T_max do deposito.

"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gurobipy as gp
from gurobipy import GRB

BASE_DIR = Path(__file__).resolve().parent.parent
# Conjunto de instancias
INSTANCE_SPECS: List[Tuple[str, Path]] = [
    ("C101", BASE_DIR / "instances" / "Solomon_C101.txt"),
    ("R101", BASE_DIR / "instances" / "Solomon_R101.txt"),
    ("RC101", BASE_DIR / "instances" / "Solomon_RC101.txt"),
]
BKS_TABLE_PATH = BASE_DIR / "data" / "solomon_bks.csv"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_TIME_LIMIT = 900


@dataclass
class InstanceData:
    name: str
    capacity: int
    num_vehicles: int
    coordinates: List[Tuple[float, float]]
    demands: List[int]
    time_windows: List[Tuple[int, int]]
    service_times: List[int]


@dataclass
class RouteResult:
    vehicle_id: int
    nodes: List[int]
    distance: float
    load: int
    start_time: float
    end_time: float


@dataclass
class InstanceResult:
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
    """Converte o .txt original de Solomon em vetores do PRVC-TW."""
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    lines_iter = iter(lines)
    name = next(lines_iter).strip()

    # Lê seção de veículos
    for line in lines_iter:
        if line.strip().upper().startswith("VEHICLE"):
            break
    for line in lines_iter:
        if line.strip().startswith("NUMBER"):
            break
    number_str, capacity_str = next(lines_iter).split()
    num_vehicles = int(number_str)
    capacity = int(capacity_str)

    # Avança para a tabela de clientes
    for line in lines_iter:
        if line.strip().upper().startswith("CUSTOMER"):
            break
    header = next(lines_iter)
    assert "CUST" in header.upper(), "Cabecalho inesperado no arquivo de Solomon."

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
        coordinates[node_id] = (float(parts[1]), float(parts[2]))
        demands[node_id] = int(parts[3])
        time_windows[node_id] = (int(parts[4]), int(parts[5]))
        service_times[node_id] = int(parts[6])

    node_ids = sorted(coordinates.keys())
    return InstanceData(
        name=name,
        capacity=capacity,
        num_vehicles=num_vehicles,
        coordinates=[coordinates[i] for i in node_ids],
        demands=[demands[i] for i in node_ids],
        time_windows=[time_windows[i] for i in node_ids],
        service_times=[service_times[i] for i in node_ids],
    )


def load_bks_table(path: Path) -> Dict[str, Dict[str, float]]:
    """Carrega a planilha com os melhores valores conhecidos (BKS)."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {
            row["instance"].strip().upper(): {
                "distance": float(row["best_known_distance"]),
                "vehicles": int(row["best_known_vehicles"]),
                "reference": row.get("bks_reference", ""),
            }
            for row in reader
        }


def attach_bks_info(result: InstanceResult, bks_row: Dict[str, float]) -> InstanceResult:
    """Atualiza o resultado com a referencia usada no comparativo."""
    result.best_known_distance = bks_row["distance"]
    result.best_known_vehicles = int(bks_row["vehicles"])
    result.distance_gap_pct = round(
        ((result.total_distance - result.best_known_distance) / result.best_known_distance) * 100.0,
        2,
    )
    result.vehicle_gap = result.vehicles_used - result.best_known_vehicles
    return result


def save_results(results: Iterable[InstanceResult], destination: Path) -> None:
    """Grava o resultado detalhado (com rotas) em JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for res in results:
        block = asdict(res)
        block["routes"] = [asdict(route) for route in res.routes]
        payload.append(block)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_summary_csv(results: Iterable[InstanceResult], destination: Path) -> None:
    """Exporta um CSV compacto para usar no relatório/slides."""
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


def build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Matriz de distancias euclidianas usada nos custos e precedencias."""
    size = len(coords)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        x_i, y_i = coords[i]
        for j in range(size):
            x_j, y_j = coords[j]
            matrix[i][j] = math.hypot(x_i - x_j, y_i - y_j)
    return matrix


def solve_instance_with_gurobi(data: InstanceData, time_limit_sec: int = DEFAULT_TIME_LIMIT) -> InstanceResult:
    """Modelo MILP com funcao objetivo, capacidade e janelas explicitas."""
    coords = data.coordinates
    distance_matrix = build_distance_matrix(coords)
    num_nodes = len(coords)
    customers = [i for i in range(num_nodes) if i != 0]
    arcs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    max_due = max(high for _, high in data.time_windows)
    max_service = max(data.service_times)
    max_travel = max(distance_matrix[i][j] for i, j in arcs)
    big_m_time = max_due + max_service + max_travel

    model = gp.Model("vrptw_gurobi")
    model.Params.TimeLimit = time_limit_sec
    model.Params.OutputFlag = 1

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    t = {
        i: model.addVar(lb=data.time_windows[i][0], ub=data.time_windows[i][1], name=f"t_{i}")
        for i in range(num_nodes)
    }
    load = {i: model.addVar(lb=0.0, ub=data.capacity, name=f"load_{i}") for i in range(num_nodes)}

    model.ModelSense = GRB.MINIMIZE
    # Funcao objetivo -> minimizar a soma dos custos c_ij * x_ij
    model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i, j in arcs))

    # (1) Atendimento unico e conservacao de fluxo
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if j != i) == 1, name=f"saida_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(num_nodes) if j != i) == 1, name=f"entrada_{i}")

    # (2) Saida/retorno ao deposito (permite desligar veiculos nao usados)
    model.addConstr(gp.quicksum(x[0, j] for j in customers) <= data.num_vehicles, name="limite_frota_saida")
    model.addConstr(gp.quicksum(x[i, 0] for i in customers) <= data.num_vehicles, name="limite_frota_entrada")

    # Precedencia temporal respeitando janelas [a_i, b_i]
    # (4) Janela de tempo global (usa T_max = due_time do deposito)
    for i, j in arcs:
        service_i = data.service_times[i]
        model.addConstr(
            t[j] >= t[i] + service_i + distance_matrix[i][j] - big_m_time * (1 - x[i, j]),
            name=f"precedencia_{i}_{j}",
        )

    model.addConstr(t[0] == data.time_windows[0][0], name="tempo_deposito")
    model.addConstr(load[0] == 0, name="carga_deposito")

    # (3) Capacidade + eliminacao de subtours (MTZ)
    for i, j in arcs:
        demand_j = data.demands[j]
        if demand_j == 0:
            continue
        model.addConstr(
            load[j] >= load[i] + demand_j - data.capacity * (1 - x[i, j]),
            name=f"capacidade_{i}_{j}",
        )

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi nao encontrou solucao viavel para {data.name} (status {model.Status}).")

    total_distance = sum(distance_matrix[i][j] * x[i, j].X for i, j in arcs)
    vehicles_used = sum(1 for j in customers if x[0, j].X > 0.5)
    runtime = model.Runtime
    routes = _extract_routes(x, t, distance_matrix, data.demands)

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


def _extract_routes(
    x_vars: Dict[Tuple[int, int], gp.Var],
    t_vars: Dict[int, gp.Var],
    distance_matrix: List[List[float]],
    demands: List[int],
) -> List[RouteResult]:
    """Reconstrói roteiros a partir das variáveis x exportando nós e cargas."""
    successors: Dict[int, int] = {}
    starts: List[int] = []
    for (i, j), var in x_vars.items():
        if var.X <= 0.5:
            continue
        successors[i] = j
        if i == 0:
            starts.append(j)

    routes: List[RouteResult] = []
    visited: set[int] = set()
    vehicle_id = 0

    for start in starts:
        if start in visited:
            continue
        route_nodes = [0]
        route_distance = 0.0
        route_load = 0
        prev = 0
        current = start
        start_time = round(float(t_vars[current].X), 2)

        while current != 0 and current not in visited:
            route_nodes.append(current)
            route_load += demands[current]
            route_distance += distance_matrix[prev][current]
            visited.add(current)
            prev = current
            current = successors.get(current, 0)

        route_nodes.append(0)
        route_distance += distance_matrix[prev][0]
        end_time = round(float(t_vars[prev].X), 2)
        routes.append(
            RouteResult(
                vehicle_id=vehicle_id,
                nodes=route_nodes,
                distance=round(route_distance, 2),
                load=route_load,
                start_time=start_time,
                end_time=end_time,
            )
        )
        vehicle_id += 1

    return routes


def main() -> None:
    """Resolve cada instancia e grava os arquivos de saida."""
    bks_table = load_bks_table(BKS_TABLE_PATH)
    collected: List[InstanceResult] = []

    for instance_name, path in INSTANCE_SPECS:
        data = parse_solomon_instance(path)
        print(f"[Gurobi] Resolvendo {instance_name} ({path}) ...")
        start = time.perf_counter()
        result = solve_instance_with_gurobi(data)
        elapsed = time.perf_counter() - start
        result.runtime_seconds = round(elapsed, 2)
        bks_info = bks_table.get(instance_name.upper())
        if bks_info:
            result = attach_bks_info(result, bks_info)
        print(
            f"  -> veiculos {result.vehicles_used}/{result.vehicles_available}, "
            f"distancia {result.total_distance:.2f} (gap {result.distance_gap_pct:.2f}%)."
        )
        collected.append(result)

    save_results(collected, RESULTS_DIR / "solomon_vrptw_gurobi.json")
    export_summary_csv(collected, RESULTS_DIR / "solomon_vrptw_gurobi.csv")


if __name__ == "__main__":
    main()

