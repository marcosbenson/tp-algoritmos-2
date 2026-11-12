"""Gráficos de los tres análisis: ruta, cobertura y comparación de criterios.

Criterios de diseño (guía de visualización de datos):
- Distancia = magnitud -> una sola escala secuencial de un tono (claro = cerca, oscuro = lejos).
- Identidad (criterios, tipo de ubicación) -> color + forma de marcador + etiqueta, nunca solo color.
- Texto siempre en tinta, no en el color de la serie; cuadrículas apenas visibles.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # sin ventana: solo guarda a archivo
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from analisis import Candidata, Quitada, distancias_a_estacion, porcentaje_cubierto
from comparacion import Resultado
from grafo import Estacion, Grafo

TINTA = "#1f2933"          # texto y estaciones
TENUE = "#6b7580"          # texto secundario y "situación actual"
CALLE = "#d5d9de"          # calles y cuadrícula: recesivas
ANTE = "#eb6834"      
VERDE = "#009b17"     # ubicaciones propuestas / ruta
VIOLETA = "#6250d6"  
ROJO =  "#ff0000"      # estaciones redundantes (otro tono y otra forma que las propuestas)
# Distancia = magnitud -> UN solo tono, de claro (cerca) a oscuro (lejos)
ESCALA = LinearSegmentedColormap.from_list("distancia", ["#dbeafe", "#1e3a8a"])
ESCALA_OLA = LinearSegmentedColormap.from_list("ola", ["#e4e7eb", "#5f6b78"])   # gris: no compite con la ruta

# Un color y una forma por criterio (primeras posiciones de la paleta categórica validada).
COLOR_CRITERIO = {"actual": "#8b9199", "cobertura": "#2a78d6", "p_mediana": "#eb6834", "p_centro": "#1baf7a"}
FORMA_CRITERIO = {"cobertura": "*", "p_mediana": "^", "p_centro": "s"}
TAMANO_CRITERIO = {"cobertura": 230, "p_mediana": 130, "p_centro": 120}


# ------------------------------------------------------------------ utilidades
def _calles(ax, grafo: Grafo, color: str = CALLE, ancho: float = 0.5) -> None:
    segmentos = []
    for u in grafo.nodos():
        lat_u, lon_u = grafo.coords(u)
        for v, _ in grafo.vecinos(u):
            if u < v or not grafo.tiene_arista(v, u):     # cada calle una sola vez
                lat_v, lon_v = grafo.coords(v)
                segmentos.append([(lon_u, lat_u), (lon_v, lat_v)])
    ax.add_collection(LineCollection(segmentos, colors=color, linewidths=ancho, zorder=1))


def _mapa_base(ax, grafo: Grafo) -> None:
    """Calles + proporción correcta (un grado de longitud mide menos que uno de latitud)."""
    _calles(ax, grafo)
    ax.autoscale()
    lats = [grafo.coords(n)[0] for n in grafo.nodos()]
    ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
    ax.set_axis_off()


def _nodos_por_distancia(ax, grafo: Grafo, dist: dict[int, float], vmax: float, cmap=ESCALA, tam: float = 5):
    orden = sorted(dist, key=dist.get)                    # los más lejanos se dibujan encima
    return ax.scatter([grafo.coords(n)[1] for n in orden], [grafo.coords(n)[0] for n in orden],
                      c=[dist[n] for n in orden], cmap=cmap, vmin=0, vmax=vmax, s=tam,
                      linewidths=0, zorder=2)


def _estaciones(ax, grafo: Grafo, nodos: set[int], tam: float = 26, etiqueta: str | None = None):
    return ax.scatter([grafo.coords(n)[1] for n in nodos], [grafo.coords(n)[0] for n in nodos],
                      s=tam, c=TINTA, edgecolors="white", linewidths=0.9, zorder=4, label=etiqueta)


def _estilo_ejes(ax, grilla: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(CALLE)
    ax.tick_params(colors=TENUE, labelsize=9)
    ax.grid(axis=grilla, color=CALLE, linewidth=0.6)
    ax.set_axisbelow(True)


def _guardar(fig, ruta_png: Path) -> None:
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _acumulada(dist: dict[int, float], pesos: dict[int, float]) -> tuple[list[float], list[float]]:
    """Curva acumulada: % de la ciudad (por peso) que queda a <= x metros de una estación."""
    total = sum(pesos.get(n, 0.0) for n in dist) or 1.0
    xs, ys, acum = [0.0], [0.0], 0.0
    for n in sorted(dist, key=dist.get):
        acum += pesos.get(n, 0.0)
        xs.append(dist[n])
        ys.append(100.0 * acum / total)
    return xs, ys


def _barras_h(ax, etiquetas: list[str], valores: list[float], color, formato: str, titulo: str) -> None:
    """Barras horizontales finas con el valor al final de cada una (sin eje x: el número está escrito).

    `color` puede ser uno solo o una lista (uno por barra)."""
    pos = list(range(len(valores)))
    ax.barh(pos, valores, height=0.62, color=color)
    tope = max(valores) if valores and max(valores) > 0 else 1.0
    ax.set_xlim(0, tope * 1.3)
    for y, v in zip(pos, valores):
        ax.text(v + tope * 0.02, y, formato.format(v), va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(pos, etiquetas, fontsize=8.5, color=TINTA)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color(CALLE)
    ax.tick_params(axis="y", length=0)
    ax.set_title(titulo, loc="left", fontsize=10.5, color=TINTA)


# ------------------------------------------------------------------ 1. ruta
def dibujar_ruta(
    grafo: Grafo,
    explorados: dict[int, float],
    camino: list[int],
    estaciones_por_nodo: dict[int, list[Estacion]],
    origen: tuple[float, float],
    destino: tuple[float, float],
    est_calle: tuple[Estacion, float] | None,
    est_recta: tuple[Estacion, float],
    metros: float | None,
    recta: float,
    ruta_png: Path,
    titulo: str,
) -> None:
    """Ruta más corta por calle, la 'ola' que exploró Dijkstra y las cifras clave."""
    fig = plt.figure(figsize=(15, 8.4), facecolor="white")
    cuadricula = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], height_ratios=[0.9, 1.2, 0.95], hspace=1.35, wspace=0.08)
    ax = fig.add_subplot(cuadricula[:, 0])

    _calles(ax, grafo)
    vmax = max(explorados.values()) if explorados else 1.0
    ola = _nodos_por_distancia(ax, grafo, explorados, vmax, ESCALA_OLA, tam=7)
    _estaciones(ax, grafo, set(estaciones_por_nodo), tam=34)

    lat_o, lon_o = origen
    lat_d, lon_d = destino
    ax.plot([lon_o, lon_d], [lat_o, lat_d], color=TENUE, linewidth=1.4, linestyle=(0, (4, 4)), zorder=3)
    if camino:
        ax.plot([grafo.coords(n)[1] for n in camino], [grafo.coords(n)[0] for n in camino],
                color=VERDE, linewidth=3.2, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    ax.scatter([lon_o], [lat_o], s=150, c="white", edgecolors=TINTA, linewidths=2.4, zorder=8, marker="o")
    ax.scatter([lon_d], [lat_d], s=150, c=TINTA, edgecolors="white", linewidths=1.6, zorder=8, marker="D")
    caja = {"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9}
    ax.annotate("Origen", (lon_o, lat_o), xytext=(10, -16), textcoords="offset points",
                fontsize=10, fontweight="bold", color=TINTA, bbox=caja, zorder=9)
    ax.annotate("Destino", (lon_d, lat_d), xytext=(10, 8), textcoords="offset points",
                fontsize=10, fontweight="bold", color=TINTA, bbox=caja, zorder=9)

    # Estación más cercana: por calle (anillo naranja) y en línea recta (anillo gris, solo si es otra).
    handles = [
        Line2D([], [], color=VERDE, linewidth=3.2, label="Ruta por calle"),
        Line2D([], [], color=TENUE, linewidth=1.4, linestyle=(0, (4, 4)), label="Línea recta"),
        Line2D([], [], marker="o", linestyle="", markersize=6, color=TINTA, label="Estaciones"),
    ]
    if est_calle is not None:
        e, _ = est_calle
        ax.scatter([e.lon], [e.lat], s=260, facecolors="none", edgecolors=VERDE, linewidths=2.4, zorder=7)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=11, markerfacecolor="none",
                              markeredgecolor=VERDE, markeredgewidth=2.2, label="Más cercana por calle"))
    er, _ = est_recta
    if est_calle is None or er.id != est_calle[0].id:
        ax.scatter([er.lon], [er.lat], s=260, facecolors="none", edgecolors=TENUE, linewidths=2.2,
                   linestyle="--", zorder=7)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=11, markerfacecolor="none",
                              markeredgecolor=TENUE, markeredgewidth=2.0, label="Más cercana en línea recta"))

    # Encuadre: lo que Dijkstra exploró (incluye origen y destino), con margen.
    puntos = [grafo.coords(n) for n in explorados] or [origen, destino]
    puntos += [origen, destino]
    lats, lons = [p[0] for p in puntos], [p[1] for p in puntos]
    mx, my = 0.06 * (max(lons) - min(lons) or 1e-3), 0.06 * (max(lats) - min(lats) or 1e-3)
    ax.set_xlim(min(lons) - mx, max(lons) + mx)
    ax.set_ylim(min(lats) - my, max(lats) + my)
    ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
    ax.set_axis_off()
    ax.set_title(titulo, loc="left", fontsize=13, color=TINTA)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, 0.0), ncol=3, frameon=False, fontsize=9.5)
    barra = fig.colorbar(ola, ax=ax, shrink=0.5, pad=0.01)
    barra.set_label("Metros por calle desde el origen (nodos que Dijkstra exploró)", fontsize=9, color=TINTA)
    barra.ax.tick_params(labelsize=8, colors=TENUE)

    # --- panel derecho: cifras -------------------------------------------------
    ax1 = fig.add_subplot(cuadricula[0, 1])
    if metros is not None:
        rodeo = metros / recta if recta else float("nan")
        _barras_h(ax1, ["Por calle", "En línea recta"], [metros, recta], [VERDE, TENUE], "{:,.0f} m",
                  f"Distancia origen → destino  ·  la calle rodea ×{rodeo:.2f}")
    else:
        ax1.set_axis_off()
        ax1.set_title("Origen → destino", loc="left", fontsize=10.5, color=TINTA)
        ax1.text(0, 0.5, "Sin camino entre esos puntos en el grafo.", fontsize=10, color=TINTA)

    ax2 = fig.add_subplot(cuadricula[1, 1])
    if est_calle is not None:
        (ec, mc), (er2, mr) = est_calle, est_recta
        _barras_h(ax2, ["Por calle", "En línea recta"], [mc, mr], [VERDE, TENUE], "{:,.0f} m",
                  "Estación más cercana al origen")
        ax2.text(0, -0.28, f"Por calle: {ec.nombre}\nEn recta: {er2.nombre}\n"
                 + ("→ coinciden" if ec.id == er2.id else "→ la calle cambia el resultado"),
                 transform=ax2.transAxes, fontsize=8.5, color=TENUE, va="top", linespacing=1.5)
    else:
        ax2.set_axis_off()
        ax2.set_title("Estación más cercana al origen", loc="left", fontsize=10.5, color=TINTA)
        ax2.text(0, 0.5, "Ninguna alcanzable por calle.", fontsize=10, color=TINTA)

    ax3 = fig.add_subplot(cuadricula[2, 1])
    ax3.set_axis_off()
    total = grafo.cantidad_nodos()
    ax3.set_title("Qué hizo Dijkstra", loc="left", fontsize=10.5, color=TINTA)
    ax3.text(0, 0.92, f"{len(explorados):,}", fontsize=26, fontweight="bold", color=TINTA, va="top")
    ax3.text(0, 0.50, f"de {total:,} nodos alcanzados antes de llegar al destino\n"
             f"({100 * len(explorados) / total:.0f} % del grafo); corta apenas fija el destino.",
             fontsize=9, color=TENUE, va="top", linespacing=1.5)
    if camino:
        ax3.text(0, 0.12, f"Ruta: {len(camino):,} nodos", fontsize=9, color=TENUE, va="top")
    _guardar(fig, ruta_png)


# ------------------------------------------------------------------ 2. cobertura
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
    invertido: Grafo | None = None,
) -> None:
    """Mapa de distancia a la estación más cercana + curva acumulada + ganancia/pérdida por ubicación."""
    quitadas = quitadas or []
    n1, n2 = len(candidatas), len(quitadas)
    filas = [4.4] + ([0.34 * n1 + 0.9] if n1 else []) + ([0.34 * n2 + 0.9] if n2 else [])
    fig = plt.figure(figsize=(15.5, max(8.2, 0.85 * sum(filas))), facecolor="white")
    cuadricula = fig.add_gridspec(len(filas), 2, width_ratios=[2.2, 1.05], height_ratios=filas,
                                  hspace=0.62, wspace=0.14)
    ax = fig.add_subplot(cuadricula[:, 0])
    ax.set_anchor("N")                       # el mapa arriba, no flotando en el medio

    # --- mapa ---------------------------------------------------------------
    _calles(ax, grafo)
    puntos = _nodos_por_distancia(ax, grafo, dist, 2 * radio)
    _estaciones(ax, grafo, nodos_estacion, etiqueta=f"Estaciones actuales ({len(nodos_estacion)})")
    caja = {"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}
    if quitadas:
        ax.scatter([q.estacion.lon for q in quitadas], [q.estacion.lat for q in quitadas],
                   s=120, marker="X", c=ROJO, edgecolors="white", linewidths=1.0, zorder=5,
                   label=f"Estaciones redundantes ({len(quitadas)})")
        for q in quitadas:
            ax.annotate(str(q.orden), (q.estacion.lon, q.estacion.lat), xytext=(7, -13),
                        textcoords="offset points", fontsize=8.5, fontweight="bold", color=TINTA,
                        bbox=caja, zorder=7)
    if candidatas:
        ax.scatter([c.lon for c in candidatas], [c.lat for c in candidatas],
                   s=240, marker="*", c=VERDE, edgecolors="white", linewidths=1.3, zorder=6,
                   label=f"Ubicaciones propuestas ({len(candidatas)})")
        for c in candidatas:
            ax.annotate(str(c.orden), (c.lon, c.lat), xytext=(7, 7), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=TINTA, bbox=caja, zorder=7)
    ax.autoscale()
    lats = [grafo.coords(n)[0] for n in dist]
    ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
    ax.set_axis_off()
    ax.set_title(titulo, loc="left", fontsize=13, color=TINTA)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=10)
    barra = fig.colorbar(puntos, ax=ax, shrink=0.55, pad=0.01)
    barra.set_label(f"Metros por calle hasta la estación más cercana (tope {2 * radio:.0f})\n"
                    f"la línea negra marca el radio de “cubierto” ({radio:.0f} m)", fontsize=9, color=TINTA)
    barra.ax.axhline(radio, color=TINTA, linewidth=1.6)           # el radio de "cubierto" en la escala
    barra.ax.tick_params(labelsize=8, colors=TENUE)

    # --- curva acumulada: qué parte de la ciudad queda a <= x metros ---------
    axc = fig.add_subplot(cuadricula[0, 1])
    xs, ys = _acumulada(dist, pesos)
    hoy = porcentaje_cubierto(dist, pesos, radio)
    axc.plot(xs, ys, color=TENUE, linewidth=2, solid_capstyle="round", label=f"Hoy ({hoy:.1f} %)")
    axc.scatter([radio], [hoy], s=64, c=TENUE, edgecolors="white", linewidths=1.6, zorder=5)
    if candidatas:
        inv = invertido if invertido is not None else grafo.invertido()
        dist2, _ = distancias_a_estacion(grafo, set(nodos_estacion) | {c.nodo for c in candidatas}, inv)
        x2, y2 = _acumulada(dist2, pesos)
        despues = porcentaje_cubierto(dist2, pesos, radio)
        axc.plot(x2, y2, color=VERDE, linewidth=2, solid_capstyle="round",
                 label=f"Con las {n1} propuestas ({despues:.1f} %)")
        axc.scatter([radio], [despues], s=64, c=VERDE, edgecolors="white", linewidths=1.6, zorder=5)
    axc.axvline(radio, color=TINTA, linewidth=1.0, zorder=1)
    axc.set_ylim(0, 100)
    axc.set_xlim(0, max(xs))
    axc.set_xlabel("Metros por calle hasta la estación más cercana", fontsize=9, color=TINTA)
    axc.set_ylabel("% de la ciudad a esa distancia o menos", fontsize=9, color=TINTA)
    axc.set_title(f"Cobertura según la distancia (línea: {radio:.0f} m)", loc="left", fontsize=10.5, color=TINTA)
    axc.legend(loc="lower right", frameon=False, fontsize=9)
    _estilo_ejes(axc)

    fila = 1
    if n1:
        axp = fig.add_subplot(cuadricula[fila, 1])
        _barras_h(axp, [f"#{c.orden}" for c in candidatas], [c.ganancia_pct for c in candidatas], VERDE,
                  "+{:.2f} pp", f"Cobertura que suma cada ubicación propuesta (a {radio:.0f} m)")
        fila += 1
    if n2:
        axq = fig.add_subplot(cuadricula[fila, 1])
        _barras_h(axq, [f"#{q.orden} {q.estacion.nombre[:22]}" for q in quitadas],
                  [q.perdida_pct for q in quitadas], ROJO, "−{:.2f} pp",
                  "Cobertura que se pierde al sacar cada estación")
    _guardar(fig, ruta_png)


# ------------------------------------------------------------------ 3. comparación
def dibujar_comparacion(
    grafo: Grafo,
    nodos_estacion: set[int],
    resultados: list[Resultado],
    radio: float,
    ruta_png: Path,
    titulo: str,
    invertido: Grafo | None = None,
) -> None:
    """Un mapa por criterio (con la distancia que resulta de aplicarlo) + una columna por métrica."""
    inv = invertido if invertido is not None else grafo.invertido()
    nuevos = [r for r in resultados if r.clave != "actual" and r.nodos]
    fig = plt.figure(figsize=(16, 10.4), facecolor="white")
    cuadricula = fig.add_gridspec(2, 12, height_ratios=[2.5, 1.0], hspace=0.22, wspace=0.9)

    # --- fila 1: un mapa por criterio, misma escala de distancia -------------
    puntos = None
    lats = [grafo.coords(n)[0] for n in grafo.nodos()]
    for i, r in enumerate(nuevos):
        ax = fig.add_subplot(cuadricula[0, 4 * i: 4 * i + 4])
        dist, _ = distancias_a_estacion(grafo, set(nodos_estacion) | set(r.nodos), inv)
        _calles(ax, grafo)
        puntos = _nodos_por_distancia(ax, grafo, dist, 2 * radio, tam=4)
        _estaciones(ax, grafo, nodos_estacion, tam=20)
        ax.scatter([grafo.coords(n)[1] for n in r.nodos], [grafo.coords(n)[0] for n in r.nodos],
                   s=TAMANO_CRITERIO[r.clave], marker=FORMA_CRITERIO[r.clave], c=COLOR_CRITERIO[r.clave],
                   edgecolors="white", linewidths=1.3, zorder=6)
        ax.autoscale()
        ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
        ax.set_axis_off()
        m = r.metricas
        ax.set_title(f"{r.nombre}\n{m['cobertura_pct']:.1f} % a ≤ {radio:.0f} m · media {m['media_m']:,.0f} m",
                     loc="left", fontsize=11, color=TINTA, linespacing=1.5)
    if puntos is not None:
        barra = fig.colorbar(puntos, ax=fig.axes[:len(nuevos)], orientation="horizontal",
                             shrink=0.35, pad=0.03, aspect=40, location="bottom")
        barra.set_label(f"Metros por calle hasta la estación más cercana, después de agregar las estaciones nuevas "
                        f"(tope {2 * radio:.0f})", fontsize=9, color=TINTA)
        barra.ax.tick_params(labelsize=8, colors=TENUE)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=TINTA,
                      label=f"Estaciones actuales ({len(nodos_estacion)})")]
    handles += [Line2D([], [], marker=FORMA_CRITERIO[r.clave], linestyle="", markersize=11,
                       markerfacecolor=COLOR_CRITERIO[r.clave], markeredgecolor="white",
                       label=f"{r.nombre}: {len(r.nodos)} nuevas") for r in nuevos]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False,
               fontsize=10)
    fig.suptitle(titulo, x=0.07, y=0.995, ha="left", fontsize=14, color=TINTA)

    # --- fila 2: una columna por métrica, una barra por criterio ------------
    metricas = [
        ("cobertura_pct", f"Cobertura a ≤ {radio:.0f} m", "más es mejor", "{:.1f} %", True),
        ("media_m", "Distancia media", "menos es mejor", "{:,.0f} m", False),
        ("p90_m", "Percentil 90", "menos es mejor", "{:,.0f} m", False),
        ("max_m", "Distancia máxima", "menos es mejor", "{:,.0f} m", False),
    ]
    nombres = {"actual": "Actual", "cobertura": "Máxima\ncobertura", "p_mediana": "p-mediana",
               "p_centro": "p-centro"}
    nuevos_todos = [r for r in resultados if r.clave != "actual"]
    for j, (clave, nombre, nota, formato, mayor) in enumerate(metricas):
        axb = fig.add_subplot(cuadricula[1, 3 * j: 3 * j + 3])
        valores = [r.metricas[clave] for r in resultados]
        mejor = max if mayor else min
        ref = mejor(r.metricas[clave] for r in nuevos_todos)
        gana = [r.clave != "actual" and abs(r.metricas[clave] - ref) <= 0.005 * abs(ref) for r in resultados]
        pos = list(range(len(resultados)))
        axb.bar(pos, valores, width=0.62, color=[COLOR_CRITERIO[r.clave] for r in resultados])
        tope = max(valores) * 1.2
        axb.set_ylim(0, tope)
        for x, v, g in zip(pos, valores, gana):
            axb.text(x, v + tope * 0.02, formato.format(v) + (" ✓" if g else ""), ha="center", va="bottom",
                     fontsize=9, color=TINTA, fontweight="bold" if g else "normal")
        axb.set_xticks(pos, [nombres[r.clave] for r in resultados], fontsize=8.5, color=TINTA)
        axb.set_yticks([])
        axb.spines[["top", "right", "left"]].set_visible(False)
        axb.spines["bottom"].set_color(CALLE)
        axb.tick_params(axis="x", length=0)
        axb.set_title(f"{nombre}  ·  {nota}", loc="left", fontsize=10.5, color=TINTA)
    fig.text(0.07, 0.035, "✓ = mejor criterio en esa métrica (se considera empate si difieren menos de 0,5 %).",
             fontsize=9, color=TENUE)
    _guardar(fig, ruta_png)
