"""Análisis de cobertura de estaciones y propuesta de ubicaciones nuevas.

Idea central: la distancia que importa es la que recorre una persona POR CALLE (respetando
sentidos) desde un punto hasta la estación más cercana. Se calcula con un solo Dijkstra
multi-fuente sobre el grafo invertido (ver `Grafo.invertido`).
"""
from __future__ import annotations

import csv
import heapq
import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

from dijkstra import dijkstra, dijkstra_multifuente
from grafo import Estacion, Grafo, haversine

DISTANCIA_MAX_DEMANDA_M = 300      # un punto de demanda a más de esto de la red se ignora


# ------------------------------------------------------------------ pesos
def pesos_por_longitud(grafo: Grafo) -> dict[int, float]:
    """Peso de cada nodo = metros de calle que salen de él.

    Sin datos de demanda, es un proxy razonable: una intersección con más calle alrededor
    representa más superficie de ciudad que un extremo de cuadra.
    """
    return {n: max(sum(w for _, w in grafo.vecinos(n)), 1.0) for n in grafo.nodos()}


def cargar_demanda(ruta: Path, grafo: Grafo) -> tuple[dict[int, float], int]:
    """Lee un CSV con columnas lat,lon[,peso] y lo asigna al nodo más cercano.

    Sirve para pasar población, comercios, estaciones de tren, universidades, etc.
    Retorna ({nodo: peso total}, cantidad de puntos ignorados por estar lejos de la red).
    """
    demanda: dict[int, float] = {}
    ignorados = 0
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            lat, lon = float(fila["lat"]), float(fila["lon"])
            peso = float(fila.get("peso") or 1.0)
            nodo = grafo.nodo_mas_cercano(lat, lon)
            if haversine(lat, lon, *grafo.coords(nodo)) > DISTANCIA_MAX_DEMANDA_M:
                ignorados += 1
                continue
            demanda[nodo] = demanda.get(nodo, 0.0) + peso
    return demanda, ignorados


# ------------------------------------------------------------- cobertura
def distancias_a_estacion(
    grafo: Grafo, nodos_estacion: Iterable[int], invertido: Grafo | None = None
) -> tuple[dict[int, float], dict[int, int]]:
    """Para cada nodo: metros por calle hasta la estación más cercana, y cuál es.

    Se corre sobre el grafo INVERTIDO porque queremos la distancia de ir hacia la estación
    (nodo -> estación), no la de salir de ella.
    """
    inv = invertido if invertido is not None else grafo.invertido()
    return dijkstra_multifuente(inv, nodos_estacion)


def porcentaje_cubierto(dist: dict[int, float], pesos: dict[int, float], umbral: float) -> float:
    """% del peso total que está a `umbral` metros o menos de una estación.

    El total se toma de `pesos` (todo el peso del grafo), no de `dist`: si `dist` es parcial
    (nodos inalcanzables, subgrafo podado), esos nodos cuentan como NO cubiertos, que es lo
    correcto. Sumar solo sobre `dist` inflaba el porcentaje.
    """
    total = sum(pesos.values())
    if total == 0:
        return 0.0
    cubierto = sum(pesos.get(n, 0.0) for n, d in dist.items() if d <= umbral)
    return 100.0 * cubierto / total


def percentil_ponderado(dist: dict[int, float], pesos: dict[int, float], q: float) -> float:
    """Distancia por debajo de la cual queda el `q` (0-1) del peso total.

    Los nodos con peso > 0 que no están en `dist` (inalcanzables) no se pueden ordenar por
    distancia: se cuentan como infinito al final. Así el percentil nunca subestima.
    """
    total = sum(pesos.values())
    if total == 0:
        return math.nan
    pares = [(d, pesos.get(n, 0.0)) for n, d in dist.items() if pesos.get(n, 0.0) > 0]
    faltan = total - sum(p for _, p in pares)
    pares.sort(key=lambda t: t[0])
    if faltan > 0:
        pares.append((math.inf, faltan))
    acumulado = 0.0
    for d, p in pares:
        acumulado += p
        if acumulado >= q * total:
            return d
    return pares[-1][0]


# -------------------------------------------------------------- espaciado
def espaciado_entre_estaciones(
    grafo: Grafo, por_nodo: dict[int, list[Estacion]]
) -> dict[str, float]:
    """Por estación: metros por calle hasta la estación más cercana (ida desde ella).

    inf si no llega a ninguna otra. 0 si comparte nodo con otra estación.

    Costo: un Dijkstra por cada nodo con estaciones (O(estaciones · (V+E) log V)). Para CABA
    (~200-400 estaciones) es aceptable; si la red crece, conviene un multi-fuente que registre
    la segunda fuente más cercana.
    """
    todos = set(por_nodo)
    resultado: dict[str, float] = {}
    for nodo, ests in por_nodo.items():
        if len(ests) > 1:
            d = 0.0
        else:
            dist, _, hallado = dijkstra(grafo, nodo, todos - {nodo})
            d = dist[hallado] if hallado is not None else math.inf
        for e in ests:
            resultado[e.id] = d
    return resultado


# -------------------------------------------------------------- propuestas
@dataclass(frozen=True)
class Candidata:
    orden: int
    nodo: int
    lat: float
    lon: float
    ganancia: float            # peso que pasa a quedar cubierto con esta estación
    ganancia_pct: float        # lo mismo como % del total
    cobertura_acumulada_pct: float
    nodos_nuevos: int          # intersecciones que pasan a quedar cubiertas


def proponer_estaciones(
    grafo: Grafo,
    dist: dict[int, float],
    pesos: dict[int, float],
    k: int,
    radio: float,
    invertido: Grafo | None = None,
) -> list[Candidata]:
    """Elige `k` ubicaciones nuevas que maximizan la cobertura (greedy de máxima cobertura).

    Un nodo está cubierto si queda a `radio` metros o menos (por calle) de una estación.
    Cada paso toma el nodo candidato que cubre más peso NO cubierto todavía; se repite k veces.
    El greedy es aproximado, pero tiene garantía conocida (>= 63 % del óptimo) porque la
    cobertura es submodular. Se implementa con evaluación perezosa (lazy greedy).

    Solo se consideran como candidatos los nodos hoy descubiertos: poner una estación donde
    ya hay una cerca no agrega cobertura.
    """
    if k <= 0:
        return []
    inv = invertido if invertido is not None else grafo.invertido()
    total = sum(pesos.values())
    cubierto_hoy = sum(pesos.get(n, 0.0) for n, d in dist.items() if d <= radio)
    descubiertos = {n for n, d in dist.items() if d > radio}
    if not descubiertos or total == 0:
        return []

    # zona[c] = nodos descubiertos desde los que se llega a c en <= radio metros
    zona: dict[int, list[int]] = {}
    for c in descubiertos:
        alcance, _, _ = dijkstra(inv, c, None, radio)
        zona[c] = [n for n in alcance if n in descubiertos]

    sin_cubrir = set(descubiertos)

    def ganancia(c: int) -> float:
        return sum(pesos.get(n, 0.0) for n in zona[c] if n in sin_cubrir)

    cola = [(-ganancia(c), c) for c in zona]
    heapq.heapify(cola)

    elegidas: list[Candidata] = []
    acumulado = cubierto_hoy
    while cola and len(elegidas) < k:
        _, c = heapq.heappop(cola)
        g = ganancia(c)                       # las cotas guardadas pueden estar desactualizadas
        if g <= 0:
            continue                          # su zona ya está cubierta; las ganancias solo bajan, se descarta
        if cola and g < -cola[0][0]:          # hay otra con cota mayor: reinsertar y seguir
            heapq.heappush(cola, (-g, c))
            continue
        nuevos = [n for n in zona[c] if n in sin_cubrir]
        sin_cubrir.difference_update(zona[c])
        acumulado += g
        lat, lon = grafo.coords(c)
        elegidas.append(Candidata(
            orden=len(elegidas) + 1, nodo=c, lat=lat, lon=lon,
            ganancia=g, ganancia_pct=100.0 * g / total,
            cobertura_acumulada_pct=100.0 * acumulado / total,
            nodos_nuevos=len(nuevos),
        ))
    return elegidas


# ------------------------------------------------------ estaciones redundantes
@dataclass(frozen=True)
class Quitada:
    orden: int
    estacion: Estacion
    perdida: float                 # peso que deja de estar cubierto al sacarla
    perdida_pct: float             # lo mismo como % del total
    cobertura_restante_pct: float
    nodos_perdidos: int            # intersecciones que quedan sin cobertura
    compartida_pct: float          # % de lo que cubre que también cubren otras (0 si no cubre nada)


def proponer_quitar(
    grafo: Grafo,
    por_nodo: dict[int, list[Estacion]],
    pesos: dict[int, float],
    cantidad: int,
    radio: float,
    invertido: Grafo | None = None,
) -> list[Quitada]:
    """Elige las `cantidad` estaciones cuya salida hace perder MENOS cobertura (greedy inverso).

    Es el problema espejo de `proponer_estaciones`: en vez de sumar la estación que más cubre,
    se saca la que menos cubre *en exclusiva*. Una intersección cubierta por 2 o más estaciones
    no se pierde al sacar una; solo cuentan las que dependen únicamente de ella.

    Al ir sacando, la pérdida de las demás solo puede crecer (otras pasan a ser la única que cubre
    un nodo), así que los valores guardados son cotas inferiores y sirve la evaluación perezosa.
    Estaciones en una misma intersección cuentan como estaciones distintas (pérdida 0 para ambas
    mientras haya otra ahí).
    """
    if not por_nodo or cantidad <= 0:
        return []
    inv = invertido if invertido is not None else grafo.invertido()
    total = sum(pesos.values())

    zona: dict[int, list[int]] = {}                 # nodo con estaciones -> nodos que llegan en <= radio
    cuenta: dict[int, int] = {}                     # nodo -> cuántas estaciones lo cubren
    for nodo, ests in por_nodo.items():
        alcance, _, _ = dijkstra(inv, nodo, None, radio)
        zona[nodo] = list(alcance)
        for n in alcance:
            cuenta[n] = cuenta.get(n, 0) + len(ests)
    cubierto = sum(pesos.get(n, 0.0) for n, c in cuenta.items() if c > 0)

    donde = {e.id: (e, nodo) for nodo, ests in por_nodo.items() for e in ests}

    def perdida(nodo: int) -> tuple[float, list[int]]:
        solos = [n for n in zona[nodo] if cuenta[n] == 1]
        return sum(pesos.get(n, 0.0) for n in solos), solos

    cola = [(perdida(nodo)[0], e.id) for nodo, ests in por_nodo.items() for e in ests]
    heapq.heapify(cola)

    quitadas: list[Quitada] = []
    restantes = len(donde)
    while cola and len(quitadas) < cantidad and restantes > 1:
        _, eid = heapq.heappop(cola)
        e, nodo = donde[eid]
        p, solos = perdida(nodo)                    # recalcular: la cota guardada puede ser vieja
        if cola and p > cola[0][0]:                 # otra podría perder menos: reinsertar
            heapq.heappush(cola, (p, eid))
            continue
        peso_zona = sum(pesos.get(n, 0.0) for n in zona[nodo])
        for n in zona[nodo]:
            cuenta[n] -= 1
        cubierto -= p
        restantes -= 1
        quitadas.append(Quitada(
            orden=len(quitadas) + 1, estacion=e, perdida=p,
            perdida_pct=100.0 * p / total if total else 0.0,
            cobertura_restante_pct=100.0 * cubierto / total if total else 0.0,
            nodos_perdidos=len(solos),
            compartida_pct=100.0 * (1.0 - p / peso_zona) if peso_zona else 0.0,
        ))
    return quitadas


def ubicaciones_de(candidatas: Collection[Candidata]) -> list[int]:
    return [c.nodo for c in candidatas]