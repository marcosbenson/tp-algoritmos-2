"""Prototipo: ruta más corta por calle y estación más cercana (por calle vs. en línea recta).

Uso:
    python main.py --demo                       # ciudad sintética, sin conexión
    python main.py --descargar                  # baja estaciones (CityBikes) y calles (OSM) y las guarda
                                                # (implica --cobertura: muestra toda la ciudad)
    python main.py                              # usa los datos guardados (ruta por defecto en CABA)
    python main.py --origen -34.6037,-58.3816 --destino -34.5810,-58.4208
    python main.py --cobertura --radio 400 --k 10   # cobertura de toda la ciudad, sin ruta puntual
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import datos
import grafico
from dijkstra import camino_minimo, dijkstra
from grafo import Estacion, Grafo, haversine

# Por defecto: Obelisco -> Plaza Italia (coordenadas aproximadas)
ORIGEN_CABA = (-34.6037, -58.3816)
DESTINO_CABA = (-34.5810, -58.4208)


def parsear_punto(texto: str) -> tuple[float, float]:
    lat, lon = (float(x) for x in texto.split(","))
    return lat, lon


def estacion_mas_cercana(
    grafo: Grafo, nodo_origen: int, por_nodo: dict[int, list[Estacion]]
) -> tuple[Estacion, float] | None:
    """Estación más cercana por calle (Dijkstra que corta en la primera estación fijada)."""
    dist, _, nodo = dijkstra(grafo, nodo_origen, por_nodo.keys())
    if nodo is None:
        return None
    return por_nodo[nodo][0], dist[nodo]


def estacion_mas_cercana_recta(lat: float, lon: float, estaciones: list[Estacion]) -> Estacion:
    return min(estaciones, key=lambda e: haversine(lat, lon, e.lat, e.lon))


def cargar_datos(args: argparse.Namespace) -> tuple[Grafo, list[Estacion]]:
    if args.demo:
        return datos.ciudad_sintetica()
    # Cachés independientes: si falta solo uno, se baja solo ese.
    if args.descargar or not datos.RUTA_ESTACIONES.exists():
        print("Descargando estaciones de CityBikes...")
        datos.guardar_estaciones(datos.descargar_estaciones())
    if args.descargar or not datos.RUTA_RED.exists():
        print("Descargando red vial de OSM (puede tardar)...")
        datos.guardar_red(datos.descargar_red_vial())
    return datos.cargar_red(), datos.cargar_estaciones()

def _mostrar_cobertura(grafo: Grafo, por_nodo: dict[int, list[Estacion]], args: argparse.Namespace) -> None:
    """Genera el mapa de cobertura de toda la ciudad (no una ruta puntual)."""
    import analisis

    inv = grafo.invertido()
    pesos = analisis.pesos_por_longitud(grafo)
    dist, _ = analisis.distancias_a_estacion(grafo, set(por_nodo), inv)
    radio = args.radio

    print(f"\n== Cobertura de toda la ciudad (radio {radio:.0f} m) ==")
    if not dist:
        print("  No hay estaciones alcanzables en la red: nada que analizar.")
        return
    for u in (300, 400, 500, 600):
        print(f"  a ≤ {u:>3} m de una estación: {analisis.porcentaje_cubierto(dist, pesos, u):5.1f} %"
              + ("  <- radio elegido" if u == round(radio) else ""))
    p50, p90, p99 = (analisis.percentil_ponderado(dist, pesos, q) for q in (0.5, 0.9, 0.99))
    print(f"  Distancia a la estación más cercana: mediana {p50:,.0f} m · p90 {p90:,.0f} m · "
          f"p99 {p99:,.0f} m · máx {max(dist.values()):,.0f} m")

    candidatas = analisis.proponer_estaciones(grafo, dist, pesos, args.k, radio, inv) if args.k > 0 else []
    if candidatas:
        print(f"\n== {len(candidatas)} ubicaciones propuestas (radio {radio:.0f} m) ==")
        print("   #   latitud    longitud   +cobertura   acumulada   intersecciones")
        for c in candidatas:
            print(f"  {c.orden:>2}  {c.lat:9.5f}  {c.lon:10.5f}   {c.ganancia_pct:6.2f} pp   "
                  f"{c.cobertura_acumulada_pct:6.1f} %   {c.nodos_nuevos:>6}")

    # Nombre de salida distinto al de analizar.py para no pisarse:
    #   main.py  -> resultados/ciudad.png
    #   analizar -> resultados/cobertura.png
    salida_png = args.png if args.png.name != "ruta.png" else Path("resultados") / "ciudad.png"
    grafico.dibujar_cobertura(
        grafo, dist, pesos, set(por_nodo), candidatas, radio, salida_png,
        f"Cobertura por calle a {radio:.0f} m · {len(por_nodo)} estaciones"
        + (f" + {len(candidatas)} propuestas" if candidatas else ""),
        [], inv,
    )
    print(f"\nGráfico de cobertura: {salida_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="usar ciudad sintética")
    ap.add_argument("--descargar", action="store_true",
                    help="forzar descarga de datos reales e mostrar la cobertura de toda la ciudad")
    ap.add_argument("--cobertura", action="store_true",
                    help="mostrar la cobertura de toda la ciudad en vez de una ruta puntual")
    ap.add_argument("--radio", type=float, default=400.0,
                    help="metros por calle para 'cubierto' (con --cobertura o --descargar)")
    ap.add_argument("--k", type=int, default=0,
                    help="estaciones nuevas a proponer en el mapa de cobertura (0 = ninguna)")
    ap.add_argument("--origen", type=parsear_punto, help="lat,lon")
    ap.add_argument("--destino", type=parsear_punto, help="lat,lon")
    ap.add_argument("--png", type=Path, default=Path("resultados") / "ruta.png",
                    help="dónde guardar el gráfico (por defecto resultados/ruta.png;\n"
                         "con --cobertura o --descargar, resultados/ciudad.png)")
    args = ap.parse_args()

    grafo, estaciones = cargar_datos(args)
    print(f"Grafo: {grafo.cantidad_nodos():,} nodos, {grafo.cantidad_aristas():,} aristas dirigidas")

    por_nodo, descartadas = datos.ubicar_estaciones(grafo, estaciones)
    usadas = len(estaciones) - len(descartadas)
    print(f"Estaciones: {usadas} de {len(estaciones)} ubicadas en {len(por_nodo)} nodos distintos")
    for e, d in descartadas[:5]:
        print(f"  ! descartada: {e.nombre} a {d:,.0f} m de la red")
    if len(descartadas) > 5:
        print(f"  ! ... y {len(descartadas) - 5} más descartadas")
    if usadas == 0:
        raise SystemExit("Ninguna estación cae sobre la red vial: revisá datos/estaciones.json.")
    if usadas < len(estaciones) / 2:
        raise SystemExit(
            "\nMás de la mitad de las estaciones están fuera de la red vial cargada.\n"
            "Probablemente son de otra ciudad. Borrá datos/estaciones.json y volvé a correr."
        )

    # --- Cobertura de toda la ciudad (implícita al descargar) -----------------
    if args.descargar or args.cobertura:
        _mostrar_cobertura(grafo, por_nodo, args)
        return

    # --- Ruta puntual (comportamiento original) -------------------------------
    if args.demo:
        origen = args.origen or (grafo.coords(5 * 60 + 5))
        destino = args.destino or (grafo.coords(54 * 60 + 54))
    else:
        origen, destino = args.origen or ORIGEN_CABA, args.destino or DESTINO_CABA

    n_o = grafo.nodo_mas_cercano(*origen)
    n_d = grafo.nodo_mas_cercano(*destino)

    t0 = time.perf_counter()
    metros, camino = camino_minimo(grafo, n_o, n_d)
    dt = time.perf_counter() - t0
    recta = haversine(*origen, *destino)
    print("\n== Ruta por calle (Dijkstra) ==")
    if not camino:
        print("No hay camino entre esos puntos en el grafo.")
    else:
        print(f"Distancia por calle: {metros:,.0f} m en {len(camino)} nodos ({dt * 1000:.1f} ms)")
        print(f"Distancia en línea recta: {recta:,.0f} m  (rodeo x{metros / recta:.2f})")

    print("\n== Estación más cercana al origen ==")
    por_calle = estacion_mas_cercana(grafo, n_o, por_nodo)
    en_recta = estacion_mas_cercana_recta(*origen, estaciones)
    if por_calle is None:
        print("Ninguna estación alcanzable por calle desde el origen.")
    else:
        est, m = por_calle
        print(f"Por calle:        {est.nombre} a {m:,.0f} m")
        d_recta = haversine(*origen, en_recta.lat, en_recta.lon)
        print(f"En línea recta:   {en_recta.nombre} a {d_recta:,.0f} m")
        print("-> " + ("coinciden" if est.id == en_recta.id else "NO coinciden: la calle cambia el resultado"))

    # Gráfico: Dijkstra con corte en el destino da los nodos explorados ("la ola") y la ruta.
    explorados, _, _ = dijkstra(grafo, n_o, [n_d])
    d_recta_est = haversine(*origen, en_recta.lat, en_recta.lon)
    grafico.dibujar_ruta(
        grafo, explorados, camino, por_nodo, origen, destino, por_calle, (en_recta, d_recta_est),
        metros if camino else None, recta, args.png,
        "Ruta más corta por calle (Dijkstra) · " + ("ciudad sintética" if args.demo else "CABA"),
    )
    print(f"\nGráfico: {args.png}")


if __name__ == "__main__":
    main()