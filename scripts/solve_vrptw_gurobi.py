"""
Solver MILP do PRVC-TW usando Gurobi nas instancias classicas de Solomon (C101, R101 e RC101).

Modelo:
  - Variaveis: x_ij em {0,1}; load_i (capacidade acumulada); t_i (tempo acumulado).
  - Funcao objetivo (OBJ):  minimizar sum_i sum_j c_ij * x_ij.
  - Restricao (1): sum_j x_ij = 1 e sum_j x_ji = 1 (cada cliente atende uma unica vez).
  - Restricao (2): sum_j x_0j = sum_i x_i0 (saida/retorno ao deposito; conservacao do fluxo).
  - Restricao (3): load_j >= load_i + d_j - Q(1 - x_ij) (capacidade e corte MTZ).
  - Restricao (4): t_j >= t_i + s_i + c_ij            (janelas individuais [a_i, b_i]).

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

# Constantes e arquivos auxiliares
BASE_DIR = Path(__file__).resolve().parent.parent
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


# Funções utilitárias
def parse_solomon_instance(path: Path) -> InstanceData:
    """Lê um arquivo .txt de Solomon e devolve vetores alinhados ao índice do nó."""
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    lines_iter = iter(lines)
    name = next(lines_iter).strip()

    # Fornece número de veículos e capacidade Q
    for line in lines_iter:
        if line.strip().upper().startswith("VEHICLE"):
            break
    for line in lines_iter:
        if line.strip().startswith("NUMBER"):
            break
    number_str, capacity_str = next(lines_iter).split()
    num_vehicles = int(number_str)
    capacity = int(capacity_str)

    # Fornece a tabela de clientes (coordenadas, demanda, janela, serviço)
    for line in lines_iter:
        if line.strip().upper().startswith("CUSTOMER"):
            break
    header = next(lines_iter)
    assert "CUST" in header.upper(), "Cabeçalho inesperado em arquivo Solomon."

    # Dicionários temporários para acumular os dados antes de transformar em listas ordenadas
    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    windows: Dict[int, Tuple[int, int]] = {}
    services: Dict[int, int] = {}

    for row in lines_iter:
        if not row.strip():
            continue
        parts = row.split()
        if len(parts) < 7:
            continue  # ignora eventuais linhas em branco
        node = int(parts[0])
        coords[node] = (float(parts[1]), float(parts[2]))
        demands[node] = int(parts[3])
        windows[node] = (int(parts[4]), int(parts[5]))
        services[node] = int(parts[6])

    ids = sorted(coords.keys())  # garante que o índice da lista corresponde ao ID do cliente
    return InstanceData(
        name=name,
        capacity=capacity,
        num_vehicles=num_vehicles,
        coordinates=[coords[i] for i in ids],
        demands=[demands[i] for i in ids],
        time_windows=[windows[i] for i in ids],
        service_times=[services[i] for i in ids],
    )


def load_bks_table(path: Path) -> Dict[str, Dict[str, float]]:
    """Lê a planilha com os melhores valores conhecidos (BKS) para comparar gaps."""
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
    """Acopla ao resultado as métricas publicadas (distância/veículos) e calcula o gap."""
    result.best_known_distance = bks_row["distance"]
    result.best_known_vehicles = int(bks_row["vehicles"])
    result.distance_gap_pct = round(
        ((result.total_distance - result.best_known_distance) / result.best_known_distance) * 100.0,
        2,
    )
    result.vehicle_gap = result.vehicles_used - result.best_known_vehicles
    return result


def save_results(results: Iterable[InstanceResult], destination: Path) -> None:
    """Grava o resultado detalhado, incluindo rotas, em formato JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for res in results:
        block = asdict(res)
        block["routes"] = [asdict(route) for route in res.routes]
        payload.append(block)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_summary_csv(results: Iterable[InstanceResult], destination: Path) -> None:
    """Gera o CSV."""
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
    """Calcula a matriz de distâncias euclidianas usada tanto no custo quanto no tempo."""
    size = len(coords)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        x_i, y_i = coords[i]
        for j in range(size):
            x_j, y_j = coords[j]
            matrix[i][j] = math.hypot(x_i - x_j, y_i - y_j)
    return matrix


# MILP 
def solve_instance_with_gurobi(data: InstanceData, time_limit_sec: int = DEFAULT_TIME_LIMIT) -> InstanceResult:
    coords = data.coordinates
    distance_matrix = build_distance_matrix(coords)
    num_nodes = len(coords)
    customers = [i for i in range(num_nodes) if i != 0]  # todos os nós exceto o depósito
    arcs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    # Big-M para a restrição (4). Usa o maior intervalo observado nas janelas dos clientes
    max_due = max(high for _, high in data.time_windows)
    max_service = max(data.service_times)
    max_travel = max(distance_matrix[i][j] for i, j in arcs)
    big_m_time = max_due + max_service + max_travel

    model = gp.Model("vrptw_gurobi")
    model.Params.TimeLimit = time_limit_sec
    model.Params.OutputFlag = 1

    # Variáveis de decisão
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    t = {
        i: model.addVar(lb=data.time_windows[i][0], ub=data.time_windows[i][1], name=f"t_{i}")
        for i in range(num_nodes)
    }
    load = {i: model.addVar(lb=0.0, ub=data.capacity, name=f"load_{i}") for i in range(num_nodes)}

    # (OBJ) minimizar a distância total percorrida 
    model.ModelSense = GRB.MINIMIZE
    model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i, j in arcs))

    # (1) Atendimento único: cada cliente deve ter exatamente uma entrada e uma saída
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if j != i) == 1, name=f"saida_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(num_nodes) if j != i) == 1, name=f"entrada_{i}")

    # (2) Conservação no depósito: número de arcos que saem do depósito = retornos
    model.addConstr(gp.quicksum(x[0, j] for j in customers) <= data.num_vehicles, name="limite_frota_saida")
    model.addConstr(gp.quicksum(x[i, 0] for i in customers) <= data.num_vehicles, name="limite_frota_entrada")

    # (4) Janelas individuais: t_j >= t_i + s_i + c_ij para cada arco viavel (i,j)
    for i, j in arcs:
        service_i = data.service_times[i]
        model.addConstr(
            t[j] >= t[i] + service_i + distance_matrix[i][j] - big_m_time * (1 - x[i, j]),
            name=f"precedencia_{i}_{j}",
        )

    # Normalização: começa no depósito, carga 0, tempo no início da janela [a_0, b_0]
    model.addConstr(t[0] == data.time_windows[0][0], name="tempo_deposito")
    model.addConstr(load[0] == 0, name="carga_deposito")

    # (3) Restrição de capacidade + eliminação de subtours (formulação MTZ em load_i)
    for i, j in arcs:
        demand_j = data.demands[j]
        if demand_j == 0:
            continue
        model.addConstr(
            load[j] >= load[i] + demand_j - data.capacity * (1 - x[i, j]),
            name=f"capacidade_{i}_{j}",
        )

    # Resolve o modelo MILP com Branch-and-Cut
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi não encontrou solução viável para {data.name} (status {model.Status}).")

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
    """Reconstrói a lista de rotas a partir dos arcos ativos (x_ij = 1)."""
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
            f"  -> veículos {result.vehicles_used}/{result.vehicles_available}, "
            f"distância {result.total_distance:.2f} (gap {result.distance_gap_pct:.2f}%)."
        )
        collected.append(result)

    save_results(collected, RESULTS_DIR / "solomon_vrptw_gurobi.json")
    export_summary_csv(collected, RESULTS_DIR / "solomon_vrptw_gurobi.csv")


if __name__ == "__main__":
    main()