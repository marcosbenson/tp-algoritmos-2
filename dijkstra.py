"""Dijkstra con cola de prioridad (heapq): O((V + E) log V)."""
from __future__ import annotations

import heapq
import math
from collections.abc import Collection

from grafo import Grafo


def dijkstra(
    grafo: Grafo,
    origen: int,
    objetivos: Collection[int] | None = None,
) -> tuple[dict[int, float], dict[int, int], int | None]:
    """Distancias mínimas desde `origen`.

    Si se pasan `objetivos`, se detiene al fijar el primero de ellos (el más cercano)
    y lo devuelve como tercer valor; sin objetivos recorre todo el grafo alcanzable.

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
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                previo[v] = u
                heapq.heappush(cola, (nd, v))
    return dist, previo, None


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
