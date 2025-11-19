from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "results" / "solomon_vrptw_gurobi.json"
FIGURES_DIR = BASE_DIR / "figures"
INSTANCE_FILES: Dict[str, Path] = {
    "C101": BASE_DIR / "instances" / "Solomon_C101.txt",
    "R101": BASE_DIR / "instances" / "Solomon_R101.txt",
    "RC101": BASE_DIR / "instances" / "Solomon_RC101.txt",
}
FIG_SIZE = (4.6, 4.6)
COORD_PADDING = 5.0

plt.style.use("seaborn-v0_8")


@dataclass
class InstanceGeometry:
    name: str
    coordinates: List[Tuple[float, float]]
    time_windows: List[Tuple[int, int]]
    service_times: List[int]


def load_instance_geometry(path: Path) -> InstanceGeometry:
    """Lê um arquivo Solomon e retorna as listas alinhadas por ID."""
    lines = [line.rstrip() for line in path.read_text().splitlines() if line.strip()]
    it = iter(lines)
    name = next(it).strip()

    while not next(it).upper().startswith("VEHICLE"):
        continue
    while not next(it).upper().startswith("CUSTOMER"):
        continue
    next(it)  # cabeçalho

    coords: Dict[int, Tuple[float, float]] = {}
    windows: Dict[int, Tuple[int, int]] = {}
    services: Dict[int, int] = {}

    for row in it:
        parts = row.split()
        if len(parts) < 7:
            continue
        node = int(parts[0])
        coords[node] = (float(parts[1]), float(parts[2]))
        windows[node] = (int(parts[4]), int(parts[5]))
        services[node] = int(parts[6])

    ordered_ids = sorted(coords.keys())
    return InstanceGeometry(
        name=name,
        coordinates=[coords[i] for i in ordered_ids],
        time_windows=[windows[i] for i in ordered_ids],
        service_times=[services[i] for i in ordered_ids],
    )


def _prepare_axes(title: str, right: float = 0.95) -> plt.Axes:
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=300)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("Coordenada X", fontsize=8)
    ax.set_ylabel("Coordenada Y", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.1, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
    fig.subplots_adjust(left=0.18, right=right, bottom=0.18, top=0.88)
    return ax


def _apply_bounds(ax: plt.Axes, geom: InstanceGeometry) -> None:
    xs = [p[0] for p in geom.coordinates]
    ys = [p[1] for p in geom.coordinates]
    ax.set_xlim(min(xs) - COORD_PADDING, max(xs) + COORD_PADDING)
    ax.set_ylim(min(ys) - COORD_PADDING, max(ys) + COORD_PADDING)


def plot_instance_graph(geom: InstanceGeometry, output_path: Path) -> None:
    depot = geom.coordinates[0]
    clients = geom.coordinates[1:]
    ax = _prepare_axes(f"Instância {geom.name} - Distribuição geográfica", right=0.78)

    xs, ys = zip(*clients)
    ax.scatter(xs, ys, c="#4a90e2", s=18, alpha=0.85, edgecolor="#0e518c", linewidth=0.2, label="Clientes")
    ax.scatter(
        depot[0],
        depot[1],
        c="#d64545",
        s=110,
        marker="s",
        edgecolor="black",
        linewidth=0.8,
        label="Depósito",
        zorder=5,
    )
    legend = ax.legend(
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=6.2,
        facecolor="#ffffff",
        edgecolor="#c0c0c0",
        framealpha=0.85,
    )
    legend.set_zorder(7)
    _apply_bounds(ax, geom)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path)
    plt.close(ax.figure)


def plot_routes(geom: InstanceGeometry, routes: Sequence[Dict], output_path: Path) -> None:
    depot = geom.coordinates[0]
    ax = _prepare_axes(f"Instância {geom.name} - Rotas obtidas", right=0.78)
    client_xs, client_ys = zip(*geom.coordinates[1:])
    ax.scatter(client_xs, client_ys, c="#d3d3d3", s=12, label="Clientes", zorder=1)
    ax.scatter(
        depot[0],
        depot[1],
        c="#111111",
        s=120,
        marker="s",
        edgecolor="white",
        linewidth=1.0,
        label="Depósito",
        zorder=6,
    )

    palette = colormaps.get_cmap("tab20").resampled(max(20, len(routes)))
    handles: List[Line2D] = []

    for idx, route in enumerate(routes):
        nodes = route["nodes"]
        coords = [geom.coordinates[node] for node in nodes]
        xs, ys = zip(*coords)
        color = palette(idx % palette.N)
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.95, zorder=2)
        ax.scatter(xs[1:-1], ys[1:-1], c=[color], s=12, edgecolor="white", linewidth=0.4, zorder=3)

        if len(xs) > 2:
            ax.text(
                xs[1],
                ys[1],
                f"V{route['vehicle_id']}",
                fontsize=6,
                weight="bold",
                color=color,
                zorder=4,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.0,
                marker="o",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=f"V{route['vehicle_id']:02d} ({route['distance']:.1f} km)",
            )
        )

    if handles:
        ncol = 2 if len(handles) > 12 else 1
        legend = ax.legend(
            handles=handles,
            frameon=True,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=6.5,
            ncol=ncol,
            facecolor="#ffffff",
            edgecolor="#c0c0c0",
            framealpha=0.9,
        )
        legend.set_zorder(7)
    _apply_bounds(ax, geom)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path)
    plt.close(ax.figure)


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    for instance_data in results:
        name = instance_data["instance"]
        path = INSTANCE_FILES.get(name)
        if not path or not path.exists():
            print(f"[plot] Arquivo da instancia {name} nao encontrado, ignorando.")
            continue
        geom = load_instance_geometry(path)
        plot_instance_graph(geom, FIGURES_DIR / f"{name}_grafo.png")
        plot_routes(geom, instance_data["routes"], FIGURES_DIR / f"{name}_rotas_publicacao.png")
        print(f"[plot] Figuras atualizadas para {name}.")


if __name__ == "__main__":
    main()
