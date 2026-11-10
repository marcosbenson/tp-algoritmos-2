"""Gráfico de cobertura: mapa de distancia a la estación más cercana + histograma."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # sin ventana: solo guarda a archivo
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from analisis import Candidata, Quitada, porcentaje_cubierto
from comparacion import Resultado
from grafo import Grafo

TINTA = "#1f2933"          # texto y estaciones
CALLE = "#d5d9de"          # calles: recesivas
ANTE = "#f97316"           # ubicaciones propuestas (se distingue de la escala azul)
VIOLETA = "#6250d6"        # estaciones redundantes (otro tono y otra forma que las propuestas)
# Distancia = magnitud -> UN solo tono, de claro (cerca) a oscuro (lejos)
ESCALA = LinearSegmentedColormap.from_list("distancia", ["#dbeafe", "#1e3a8a"])


def dibujar_cobertura(
    grafo: Grafo,
    dist: dict[int, float],
    pesos: dict[int, float],
    nodos_estacion: set[int],
    candidatas: list[Candidata],
    radio: float,
    ruta_png: Path,
    titulo: str,
    quitadas: list[Quitada] | None = None,
) -> None:
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [2.3, 1]}, facecolor="white"
    )

    # --- mapa --------------------------------------------------------------
    segmentos = []
    for u in grafo.nodos():
        lat_u, lon_u = grafo.coords(u)
        for v, _ in grafo.vecinos(u):
            if u < v or not grafo.tiene_arista(v, u):     # cada calle una sola vez
                lat_v, lon_v = grafo.coords(v)
                segmentos.append([(lon_u, lat_u), (lon_v, lat_v)])
    ax.add_collection(LineCollection(segmentos, colors=CALLE, linewidths=0.5, zorder=1))

    orden = sorted(dist, key=dist.get)                    # los más lejanos se dibujan encima
    lats = [grafo.coords(n)[0] for n in orden]
    lons = [grafo.coords(n)[1] for n in orden]
    puntos = ax.scatter(lons, lats, c=[dist[n] for n in orden], cmap=ESCALA,
                        vmin=0, vmax=2 * radio, s=5, linewidths=0, zorder=2)

    ax.scatter([grafo.coords(n)[1] for n in nodos_estacion],
               [grafo.coords(n)[0] for n in nodos_estacion],
               s=26, c=TINTA, edgecolors="white", linewidths=0.9, zorder=4,
               label=f"Estaciones actuales ({len(nodos_estacion)})")
    if quitadas:
        ax.scatter([q.estacion.lon for q in quitadas], [q.estacion.lat for q in quitadas],
                   s=110, marker="X", c=VIOLETA, edgecolors="white", linewidths=1.0, zorder=5,
                   label=f"Estaciones redundantes ({len(quitadas)})")
    if candidatas:
        ax.scatter([c.lon for c in candidatas], [c.lat for c in candidatas],
                   s=230, marker="*", c=ANTE, edgecolors="white", linewidths=1.3, zorder=6,
                   label=f"Ubicaciones propuestas ({len(candidatas)})")
        for c in candidatas:
            ax.annotate(str(c.orden), (c.lon, c.lat), xytext=(7, 7), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=TINTA, zorder=7,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})

    ax.autoscale()
    ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
    ax.set_axis_off()
    ax.set_title(titulo, loc="left", fontsize=13, color=TINTA)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=10)
    barra = fig.colorbar(puntos, ax=ax, shrink=0.55, pad=0.01)
    barra.set_label(f"Metros por calle hasta la estación más cercana (tope {2 * radio:.0f})",
                    fontsize=9, color=TINTA)

    # --- histograma --------------------------------------------------------
    total = sum(pesos.get(n, 0.0) for n in dist)
    valores = list(dist.values())
    anchos = [100.0 * pesos.get(n, 0.0) / total for n in dist] if total else None
    tope = max(valores)                       # sin recortar: la cola larga es información
    axh.hist(valores, bins=30, range=(0, tope), weights=anchos,
             color="#2563eb", edgecolor="white", linewidth=0.6)
    axh.axvline(radio, color=TINTA, linestyle="--", linewidth=1.2)
    pct = porcentaje_cubierto(dist, pesos, radio)
    axh.text(0.98, 0.97, f"{pct:.0f} % de la ciudad a ≤ {radio:.0f} m\n(línea punteada)",
             transform=axh.transAxes, ha="right", va="top", fontsize=10, color=TINTA)
    axh.set_xlabel("Metros por calle hasta la estación más cercana", fontsize=9, color=TINTA)
    axh.set_ylabel("% de la ciudad (por metros de calle)", fontsize=9, color=TINTA)
    axh.set_title("Distribución de la distancia", loc="left", fontsize=12, color=TINTA)
    axh.spines[["top", "right"]].set_visible(False)
    axh.grid(axis="y", color=CALLE, linewidth=0.6)
    axh.set_axisbelow(True)

    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Un color y una forma por criterio (primeras 3 posiciones de la paleta categórica validada).
# La forma distinta + los valores escritos en las barras hacen que no dependa solo del color.
COLOR_CRITERIO = {"actual": "#8b9199", "cobertura": "#2a78d6", "p_mediana": "#eb6834", "p_centro": "#1baf7a"}
FORMA_CRITERIO = {"cobertura": "*", "p_mediana": "^", "p_centro": "s"}
TAMANO_CRITERIO = {"cobertura": 210, "p_mediana": 120, "p_centro": 110}


def dibujar_comparacion(
    grafo: Grafo,
    nodos_estacion: set[int],
    resultados: list[Resultado],
    radio: float,
    ruta_png: Path,
    titulo: str,
) -> None:
    """Mapa con las ubicaciones de cada criterio + una barra por métrica (cada una en su unidad)."""
    fig = plt.figure(figsize=(15.5, 8), facecolor="white")
    cuadricula = fig.add_gridspec(4, 2, width_ratios=[2.0, 1.1], hspace=0.95, wspace=0.12)
    ax = fig.add_subplot(cuadricula[:, 0])

    segmentos = []
    for u in grafo.nodos():
        lat_u, lon_u = grafo.coords(u)
        for v, _ in grafo.vecinos(u):
            if u < v or not grafo.tiene_arista(v, u):
                lat_v, lon_v = grafo.coords(v)
                segmentos.append([(lon_u, lat_u), (lon_v, lat_v)])
    ax.add_collection(LineCollection(segmentos, colors=CALLE, linewidths=0.5, zorder=1))
    lats = [grafo.coords(n)[0] for n in grafo.nodos()]
    ax.scatter([grafo.coords(n)[1] for n in nodos_estacion], [grafo.coords(n)[0] for n in nodos_estacion],
               s=24, c=TINTA, edgecolors="white", linewidths=0.8, zorder=3,
               label=f"Estaciones actuales ({len(nodos_estacion)})")
    for r in resultados:
        if r.clave == "actual" or not r.nodos:
            continue
        ax.scatter([grafo.coords(n)[1] for n in r.nodos], [grafo.coords(n)[0] for n in r.nodos],
                   s=TAMANO_CRITERIO[r.clave], marker=FORMA_CRITERIO[r.clave], c=COLOR_CRITERIO[r.clave],
                   edgecolors="white", linewidths=1.2, zorder=5, label=f"{r.nombre} ({len(r.nodos)})")
    ax.autoscale()
    ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
    ax.set_axis_off()
    ax.set_title(titulo, loc="left", fontsize=13, color=TINTA)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=10)

    metricas = [
        ("cobertura_pct", f"Cobertura a ≤ {radio:.0f} m (%)  ·  más es mejor", "{:.1f} %"),
        ("media_m", "Distancia media (m)  ·  menos es mejor", "{:,.0f} m"),
        ("p90_m", "Percentil 90 (m)  ·  menos es mejor", "{:,.0f} m"),
        ("max_m", "Distancia máxima (m)  ·  menos es mejor", "{:,.0f} m"),
    ]
    for i, (clave, titulo_m, formato) in enumerate(metricas):
        axb = fig.add_subplot(cuadricula[i, 1])
        valores = [r.metricas[clave] for r in resultados]
        posiciones = list(range(len(resultados)))
        axb.barh(posiciones, valores, height=0.62, color=[COLOR_CRITERIO[r.clave] for r in resultados])
        tope = max(valores) * 1.28
        axb.set_xlim(0, tope)
        for y, v in zip(posiciones, valores):
            axb.text(v + tope * 0.015, y, formato.format(v), va="center", fontsize=9, color=TINTA)
        axb.set_yticks(posiciones, [r.nombre for r in resultados], fontsize=9, color=TINTA)
        axb.invert_yaxis()
        axb.set_xticks([])
        axb.spines[["top", "right", "bottom"]].set_visible(False)
        axb.spines["left"].set_color(CALLE)
        axb.tick_params(axis="y", length=0)
        axb.set_title(titulo_m, loc="left", fontsize=10, color=TINTA)

    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
