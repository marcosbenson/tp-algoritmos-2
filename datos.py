"""Obtención de datos: estaciones (CityBikes), red vial (OpenStreetMap) y ciudad sintética."""
from __future__ import annotations

import gzip
import json
import math
import random
import time
from pathlib import Path

from grafo import Estacion, Grafo, haversine

CARPETA = Path(__file__).parent / "datos"
RUTA_ESTACIONES = CARPETA / "estaciones.json"
RUTA_RED = CARPETA / "red_vial.json.gz"

API_CITYBIKES = "https://api.citybik.es/v2"
CIUDAD = "buenos aires"                  # se busca la red por ciudad y país, NO por id fijo
PAIS = "AR"                              # (el id "ecobici" es el de Ciudad de México)
LUGAR_OSM = "Ciudad Autónoma de Buenos Aires, Argentina"
DISTANCIA_MAX_NODO_M = 300               # una estación más lejos que esto de la red se descarta
# Servidores públicos de Overpass (se prueban en orden; sin "/interpreter" al final)
ENDPOINTS_OVERPASS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
REINTENTOS = 3                           # por servidor, con espera doble entre intentos
TIMEOUT_OSMNX_S = 180                    # timeout por request de Overpass (osmnx)


# ---------------------------------------------------------------- estaciones
def descargar_estaciones(ciudad: str = CIUDAD, pais: str = PAIS) -> list[Estacion]:
    """Baja las estaciones de CityBikes de `ciudad`. Solo usa nombre y ubicación."""
    import requests

    redes = requests.get(f"{API_CITYBIKES}/networks", timeout=30).json()["networks"]
    candidatas = [
        r for r in redes
        if ciudad in r["location"]["city"].lower() and r["location"].get("country") == pais
    ]
    if not candidatas:
        raise RuntimeError(f"No encontré ninguna red de {ciudad!r} ({pais}) en CityBikes")
    if len(candidatas) > 1:
        print("  Varias redes candidatas:", [r["id"] for r in candidatas], "-> uso la primera")
    resp = requests.get(f"{API_CITYBIKES}/networks/{candidatas[0]['id']}", timeout=30)
    resp.raise_for_status()

    estaciones = []
    for s in resp.json()["network"]["stations"]:
        if s.get("latitude") is None or s.get("longitude") is None:
            continue
        estaciones.append(Estacion(str(s["id"]), s["name"], s["latitude"], s["longitude"]))
    return estaciones


def guardar_estaciones(estaciones: list[Estacion], ruta: Path = RUTA_ESTACIONES) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps([e.__dict__ for e in estaciones], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def cargar_estaciones(ruta: Path = RUTA_ESTACIONES) -> list[Estacion]:
    return [Estacion(**e) for e in json.loads(ruta.read_text(encoding="utf-8"))]


# ----------------------------------------------------------------- red vial
def _configurar_osmnx(endpoint: str) -> None:
    """Apunta osmnx al endpoint de Overpass y fija timeout/rate limit de forma compatible.

    El nombre del setting cambió entre versiones de osmnx; se prueban los dos.
    """
    import osmnx as ox

    for attr in ("overpass_url", "overpass_endpoint"):
        if hasattr(ox.settings, attr):
            setattr(ox.settings, attr, endpoint)
    if hasattr(ox.settings, "timeout"):
        ox.settings.timeout = TIMEOUT_OSMNX_S
    if hasattr(ox.settings, "overpass_rate_limit"):
        ox.settings.overpass_rate_limit = False   # los reintentos con backoff ya lo manejan


def descargar_red_vial(lugar: str = LUGAR_OSM) -> Grafo:
    """Baja la red de calles aptas para bici de OSM y la convierte a nuestro `Grafo`."""
    import osmnx as ox                   # import perezoso: solo se necesita al descargar

    g_osm = None
    for endpoint in ENDPOINTS_OVERPASS:
        _configurar_osmnx(endpoint)
        espera = 10
        for intento in range(1, REINTENTOS + 1):
            try:
                print(f"  Overpass: {endpoint} (intento {intento}/{REINTENTOS})")
                g_osm = ox.graph_from_place(lugar, network_type="bike")
                break
            except Exception as exc:                          # 502/504/timeout: servidor saturado
                print(f"    falló: {str(exc).splitlines()[0][:110]}")
                if intento < REINTENTOS:
                    print(f"    reintento en {espera} s...")
                    time.sleep(espera)
                    espera *= 2
        if g_osm is not None:
            break
    if g_osm is None:
        raise RuntimeError("Ningún servidor Overpass respondió. Probá de nuevo en unos minutos.")
    grafo = Grafo()
    for nodo, d in g_osm.nodes(data=True):
        grafo.agregar_nodo(int(nodo), d["y"], d["x"])
    for u, v, d in g_osm.edges(data=True):
        grafo.agregar_arista(int(u), int(v), float(d["length"]))
    return grafo


def guardar_red(grafo: Grafo, ruta: Path = RUTA_RED) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wt", encoding="utf-8") as f:
        json.dump(grafo.a_dict(), f)


def cargar_red(ruta: Path = RUTA_RED) -> Grafo:
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return Grafo.desde_dict(json.load(f))


# ------------------------------------------------------- estaciones en el grafo
def ubicar_estaciones(
    grafo: Grafo, estaciones: list[Estacion]
) -> tuple[dict[int, list[Estacion]], list[tuple[Estacion, float]]]:
    """Asocia cada estación a su nodo más cercano.

    Retorna ({nodo: [estaciones]}, [(estación, metros al nodo)] de las descartadas).
    Se descartan las estaciones a más de DISTANCIA_MAX_NODO_M de la red (están fuera
    de la zona cargada o tienen mal las coordenadas): asociarlas igual daría rutas falsas.
    Dos estaciones pueden caer en el mismo nodo; por eso es una lista por nodo.
    """
    por_nodo: dict[int, list[Estacion]] = {}
    descartadas: list[tuple[Estacion, float]] = []
    for e in estaciones:
        nodo = grafo.nodo_mas_cercano(e.lat, e.lon)
        lat, lon = grafo.coords(nodo)
        d = haversine(e.lat, e.lon, lat, lon)
        if d > DISTANCIA_MAX_NODO_M:
            descartadas.append((e, d))
        else:
            por_nodo.setdefault(nodo, []).append(e)
    return por_nodo, descartadas


# ---------------------------------------------------------- ciudad sintética
def ciudad_sintetica(
    filas: int = 60, columnas: int = 60, paso_m: float = 80.0, semilla: int = 42
) -> tuple[Grafo, list[Estacion]]:
    """Grilla de calles cerca del Obelisco, para probar sin conexión.

    Incluye: ~10 % de calles de una sola mano, ~6 % de tramos cortados y una
    'autopista' vertical que solo se cruza por 3 puentes (para que la estación más
    cercana en línea recta no siempre sea la más cercana por calle).
    """
    rnd = random.Random(semilla)
    lat0, lon0 = -34.6037, -58.3816
    dlat = paso_m / 111_320
    dlon = paso_m / (111_320 * math.cos(math.radians(lat0)))

    g = Grafo()
    id_nodo = lambda f, c: f * columnas + c   # noqa: E731
    for f in range(filas):
        for c in range(columnas):
            g.agregar_nodo(id_nodo(f, c), lat0 + f * dlat, lon0 + c * dlon)

    col_barrera, puentes = columnas // 2, {8, filas // 2, filas - 9}
    for f in range(filas):
        for c in range(columnas):
            vecinos = []
            if c + 1 < columnas:
                vecinos.append((f, c + 1))
            if f + 1 < filas:
                vecinos.append((f + 1, c))
            for f2, c2 in vecinos:
                cruza_barrera = c == col_barrera - 1 and c2 == col_barrera and f not in puentes
                if cruza_barrera or rnd.random() < 0.06:
                    continue
                u, v = id_nodo(f, c), id_nodo(f2, c2)
                d = haversine(*g.coords(u), *g.coords(v))
                if rnd.random() < 0.10:               # calle de una mano
                    if rnd.random() < 0.5:
                        u, v = v, u
                    g.agregar_arista(u, v, d)
                else:
                    g.agregar_arista(u, v, d)
                    g.agregar_arista(v, u, d)

    estaciones = []
    for i in range(25):
        nodo = rnd.randrange(filas * columnas)
        lat, lon = g.coords(nodo)
        estaciones.append(Estacion(f"s{i:02d}", f"Estación sintética {i:02d}",
                                   lat + rnd.uniform(-2e-4, 2e-4), lon + rnd.uniform(-2e-4, 2e-4)))
    return g, estaciones