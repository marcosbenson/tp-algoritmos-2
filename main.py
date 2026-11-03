"""Prototipo: ruta más corta por calle y estación más cercana (por calle vs. en línea recta).

Uso:
    python main.py --demo                       # ciudad sintética, sin conexión
    python main.py --descargar                  # baja estaciones (CityBikes) y calles (OSM) y las guarda
    python main.py                              # usa los datos guardados (ruta por defecto en CABA)
    python main.py --origen=-34.6037,-58.3816 --destino=-34.5810,-58.4208
"""
from __future__ import annotations

import argparse
import time

import datos
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="usar ciudad sintética")
    ap.add_argument("--descargar", action="store_true", help="forzar descarga de datos reales")
    ap.add_argument("--origen", type=parsear_punto, help="lat,lon")
    ap.add_argument("--destino", type=parsear_punto, help="lat,lon")
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
    if usadas < len(estaciones) / 2:
        raise SystemExit(
            "\nMás de la mitad de las estaciones están fuera de la red vial cargada.\n"
            "Probablemente son de otra ciudad. Borrá datos/estaciones.json y volvé a correr."
        )

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


if __name__ == "__main__":
    main()
