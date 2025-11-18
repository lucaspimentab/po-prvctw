"""
Modelo MILP do PRVC-TW resolvido com Gurobi para as instancias de Solomon.

O codigo explicita funcao objetivo, variaveis e restricoes para fins de
Documentacao na disciplina de Pesquisa Operacional.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solve_vrptw import (
    INSTANCE_SPECS,
    BKS_TABLE_PATH,
    RESULTS_DIR,
    InstanceData,
    InstanceResult,
    RouteResult,
    attach_bks_info,
    export_summary_csv,
    load_bks_table,
    parse_solomon_instance,
    save_results,
)

DEFAULT_TIME_LIMIT = 900  # 15 minutos por instancia costuma ser suficiente


def build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Distancias euclidianas usadas como custo e tempo de deslocamento."""
    size = len(coords)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        x_i, y_i = coords[i]
        for j in range(size):
            x_j, y_j = coords[j]
            matrix[i][j] = math.hypot(x_i - x_j, y_i - y_j)
    return matrix


def solve_instance_with_gurobi(data: InstanceData, time_limit_sec: int = DEFAULT_TIME_LIMIT) -> InstanceResult:
    """Resolve a instancia com um MILP classico do PRVC-TW."""
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

    # Variaveis de decisao
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")  # 1 se o arco (i,j) e percorrido
    t = {
        i: model.addVar(lb=data.time_windows[i][0], ub=data.time_windows[i][1], name=f"t_{i}")
        for i in range(num_nodes)
    }  # inicio do atendimento
    load = {i: model.addVar(lb=0.0, ub=data.capacity, name=f"load_{i}") for i in range(num_nodes)}

    model.ModelSense = GRB.MINIMIZE
    model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i, j] for i, j in arcs))

    # Cada cliente precisa entrar e sair exatamente uma vez
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if j != i) == 1, name=f"saida_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(num_nodes) if j != i) == 1, name=f"entrada_{i}")

    # Limite do numero de veiculos (cada arco que sai/entra do deposito conta)
    model.addConstr(gp.quicksum(x[0, j] for j in customers) <= data.num_vehicles, name="limite_frota_saida")
    model.addConstr(gp.quicksum(x[i, 0] for i in customers) <= data.num_vehicles, name="limite_frota_entrada")

    # Propagacao temporal respeitando janelas [a_i, b_i]
    for i, j in arcs:
        service_i = data.service_times[i]
        model.addConstr(
            t[j] >= t[i] + service_i + distance_matrix[i][j] - big_m_time * (1 - x[i, j]),
            name=f"precedencia_{i}_{j}",
        )

    # Normalizacao no deposito
    model.addConstr(t[0] == data.time_windows[0][0], name="tempo_deposito")
    model.addConstr(load[0] == 0, name="carga_deposito")

    # Restricao de capacidade baseada em fluxo
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
    x_vars: Dict[Tuple[int, int], gp.Var], t_vars: Dict[int, gp.Var], distance_matrix: List[List[float]], demands: List[int]
) -> List[RouteResult]:
    """Reconstrucao das rotas a partir das variaveis binarias x."""
    num_nodes = len(demands)
    successors: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for (i, j), var in x_vars.items():
        if var.X > 0.5:
            successors[i].append(j)

    routes: List[RouteResult] = []
    vehicle_id = 0
    for start in list(successors[0]):
        if start == 0:
            continue
        route_nodes = [0]
        route_distance = 0.0
        route_load = 0
        prev = 0
        current = start
        start_time = round(t_vars[current].X, 2)

        while True:
            route_nodes.append(current)
            route_load += demands[current]
            route_distance += distance_matrix[prev][current]
            next_nodes = successors.get(current, [])
            prev = current
            if not next_nodes:
                current = 0
            else:
                current = next_nodes[0]
            if current == 0:
                break

        route_nodes.append(0)
        route_distance += distance_matrix[prev][0]
        end_time = round(t_vars[prev].X, 2)
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
    bks = load_bks_table(BKS_TABLE_PATH)
    results: List[InstanceResult] = []

    for instance_name, path in INSTANCE_SPECS:
        data = parse_solomon_instance(path)
        print(f"[Gurobi] Resolvendo {instance_name} ({path}) ...")
        start = time.perf_counter()
        result = solve_instance_with_gurobi(data)
        elapsed = time.perf_counter() - start
        result.runtime_seconds = round(elapsed, 2)
        bks_info = bks.get(instance_name.upper())
        if bks_info:
            result = attach_bks_info(result, bks_info)
        print(
            f"  -> veiculos {result.vehicles_used}/{result.vehicles_available}, "
            f"distancia {result.total_distance:.2f}, gap {result.distance_gap_pct:.2f}%."
        )
        results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(results, RESULTS_DIR / "solomon_vrptw_gurobi.json")
    export_summary_csv(results, RESULTS_DIR / "solomon_vrptw_gurobi.csv")


if __name__ == "__main__":
    main()
