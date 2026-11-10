"""Dijkstra con cola de prioridad (heapq): O((V + E) log V)."""
from __future__ import annotations

import heapq
import math
from collections.abc import Collection, Iterable

from grafo import Grafo


def dijkstra(
    grafo: Grafo,
    origen: int,
    objetivos: Collection[int] | None = None,
    radio_max: float | None = None,
) -> tuple[dict[int, float], dict[int, int], int | None]:
    """Distancias mínimas desde `origen`.

    Si se pasan `objetivos`, se detiene al fijar el primero de ellos (el más cercano)
    y lo devuelve como tercer valor; sin objetivos recorre todo el grafo alcanzable.
    Con `radio_max`, ignora todo lo que quede a más de esa distancia (búsqueda acotada).

    Retorna (distancias, previo, objetivo_alcanzado).
    """
    objetivos_set = set(objetivos) if objetivos is not None else None
    dist: dict[int, float] = {origen: 0.0}
    previo: dict[int, int] = {}
    fijados: set[int] = set()
    cola: list[tuple[float, int]] = [(0.0, origen)]

    while cola:
        d, u = heapq.heappop(cola)
        if u in fijados:          # entrada vieja de la cola (había una mejor)
            continue
        fijados.add(u)
        if objetivos_set is not None and u in objetivos_set:
            return dist, previo, u
        for v, peso in grafo.vecinos(u):
            if v in fijados:
                continue
            nd = d + peso
            if radio_max is not None and nd > radio_max:
                continue
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                previo[v] = u
                heapq.heappush(cola, (nd, v))
    return dist, previo, None


def dijkstra_multifuente(
    grafo: Grafo,
    fuentes: Iterable[int],
    radio_max: float | None = None,
) -> tuple[dict[int, float], dict[int, int]]:
    """Dijkstra desde varias fuentes a la vez (todas con distancia 0 al arrancar).

    Una sola pasada da, para cada nodo, la distancia a la fuente MÁS CERCANA y cuál es,
    en vez de correr un Dijkstra por fuente. Mismo costo que uno solo: O((V + E) log V).

    Retorna (distancia, fuente_mas_cercana).
    """
    dist: dict[int, float] = {}
    cercana: dict[int, int] = {}
    cola: list[tuple[float, int]] = []
    for f in fuentes:
        if f not in dist:
            dist[f] = 0.0
            cercana[f] = f
            heapq.heappush(cola, (0.0, f))

    fijados: set[int] = set()
    while cola:
        d, u = heapq.heappop(cola)
        if u in fijados:
            continue
        fijados.add(u)
        for v, peso in grafo.vecinos(u):
            if v in fijados:
                continue
            nd = d + peso
            if radio_max is not None and nd > radio_max:
                continue
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                cercana[v] = cercana[u]
                heapq.heappush(cola, (nd, v))
    return dist, cercana


def dijkstra_podado(
    grafo: Grafo, origen: int, actual: dict[int, float]
) -> dict[int, float]:
    """Nodos que MEJORAN su distancia `actual` si `origen` pasara a ser una fuente.

    Es un Dijkstra desde `origen` que no se expande por nodos donde ya hay algo igual o más
    cercano. La poda es exacta: si llegar a v desde `origen` no mejora actual[v], tampoco
    mejora ningún nodo más allá de v, porque actual[n] <= d(n, v) + actual[v] (desigualdad
    triangular). Por eso explora solo la zona donde `origen` realmente ayudaría.

    `grafo` debe ser el INVERTIDO si `actual` mide distancias hacia las estaciones.
    Retorna {nodo: nueva distancia}, incluyendo a `origen` con 0 (vacío si ya es fuente).
    """
    if not 0.0 < actual.get(origen, math.inf):
        return {}
    mejor: dict[int, float] = {origen: 0.0}
    fijados: set[int] = set()
    cola: list[tuple[float, int]] = [(0.0, origen)]
    while cola:
        d, u = heapq.heappop(cola)
        if u in fijados:
            continue
        fijados.add(u)
        for v, peso in grafo.vecinos(u):
            nd = d + peso
            if v not in fijados and nd < actual.get(v, math.inf) and nd < mejor.get(v, math.inf):
                mejor[v] = nd
                heapq.heappush(cola, (nd, v))
    return mejor


def reconstruir_camino(previo: dict[int, int], origen: int, destino: int) -> list[int]:
    """Camino origen -> destino siguiendo los `previo`. Lista vacía si no hay camino."""
    if destino != origen and destino not in previo:
        return []
    camino = [destino]
    while camino[-1] != origen:
        camino.append(previo[camino[-1]])
    camino.reverse()
    return camino


def camino_minimo(grafo: Grafo, origen: int, destino: int) -> tuple[float, list[int]]:
    """(distancia, camino) de origen a destino; (inf, []) si no hay camino."""
    dist, previo, alcanzado = dijkstra(grafo, origen, {destino})
    if alcanzado is None:
        return math.inf, []
    return dist[destino], reconstruir_camino(previo, origen, destino)
