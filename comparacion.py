"""Tres criterios para ubicar estaciones nuevas, evaluados con las mismas métricas.

  * Máxima cobertura: maximizar el % de la ciudad a <= `radio` metros de una estación.
  * p-mediana:        minimizar la distancia MEDIA (ponderada) a la estación más cercana.
  * p-centro:         minimizar la distancia MÁXIMA a la estación más cercana.

Los tres parten de las estaciones actuales y agregan `k` nuevas, y se evalúan igual: se recalcula
todo desde cero con un Dijkstra multi-fuente, así ninguno se juzga con su propia vara.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from analisis import (
    distancias_a_estacion,
    percentil_ponderado,
    porcentaje_cubierto,
    proponer_estaciones,
)
from dijkstra import dijkstra_podado
from grafo import Grafo, haversine

ACTUAL, COBERTURA, MEDIANA, CENTRO = "actual", "cobertura", "p_mediana", "p_centro"
NOMBRES = {
    ACTUAL: "Actual",
    COBERTURA: "Máxima cobertura",
    MEDIANA: "p-mediana",
    CENTRO: "p-centro",
}


@dataclass(frozen=True)
class Resultado:
    clave: str
    nombre: str
    nodos: list[int]                 # ubicaciones nuevas, en orden de elección
    metricas: dict[str, float]


# ------------------------------------------------------------- evaluación
def evaluar(
    grafo: Grafo,
    invertido: Grafo,
    nodos_estacion: set[int],
    pesos: dict[int, float],
    radio: float,
) -> dict[str, float]:
    """Métricas de un conjunto de estaciones, calculadas desde cero.

    Solo cuentan los nodos con peso > 0 (con datos de demanda, los lugares sin demanda no
    deben decidir la distancia máxima). Si no hay ningún nodo con peso, las métricas que
    dividen por el total devuelven NaN (no hay población a la cual promediar).
    """
    dist, _ = distancias_a_estacion(grafo, nodos_estacion, invertido)
    activos = {n: d for n, d in dist.items() if pesos.get(n, 0.0) > 0}
    total = sum(pesos[n] for n in activos)
    if total == 0:
        return {"cobertura_pct": 0.0, "media_m": math.nan, "p90_m": math.nan, "max_m": math.nan}
    return {
        "cobertura_pct": porcentaje_cubierto(activos, pesos, radio),
        "media_m": sum(pesos[n] * d for n, d in activos.items()) / total,
        "p90_m": percentil_ponderado(activos, pesos, 0.9),
        "max_m": max(activos.values()),
    }


# --------------------------------------------------------------- p-mediana
def p_mediana(
    grafo: Grafo, dist: dict[int, float], pesos: dict[int, float], k: int, invertido: Grafo | None = None
) -> list[int]:
    """Greedy para p-mediana: cada paso agrega la estación que más baja la suma de peso*distancia.

    Ganancia de un candidato c = suma sobre los nodos que mejorarían de peso * (distancia actual
    - distancia nueva). Esa ganancia solo puede bajar a medida que se agregan estaciones
    (función submodular), así que las cotas guardadas en la cola son superiores y nunca
    subestiman. En cada pop se recalcula la ganancia real del candidato sacado (con
    `dijkstra_podado`, que es barato porque solo explora donde ese candidato ayudaría);
    si resulta menor que la cota del tope, se reinserta con la cota actualizada. Eso no es
    el lazy greedy "puro" (que evita recalcular hasta que el tope gana), pero el podado hace
    que el recálculo sea económico y el resultado es el mismo.

    Requiere que `dist` cubra todos los nodos (red fuertemente conexa con >= 1 estación).
    """
    if k <= 0:
        return []
    inv = invertido if invertido is not None else grafo.invertido()
    actual = dict(dist)

    def ganancia(c: int) -> float:
        return sum(pesos.get(n, 0.0) * (actual[n] - nd) for n, nd in dijkstra_podado(inv, c, actual).items())

    # Candidatos: todos los nodos con distancia > 0. Los que no tienen peso ni vecinos con peso
    # dan ganancia 0 y se descartan solos (el tope con ganancia <= 0 corta la búsqueda).
    cola = [(-ganancia(c), c) for c, d in actual.items() if d > 0]
    heapq.heapify(cola)

    elegidas: list[int] = []
    while cola and len(elegidas) < k:
        _, c = heapq.heappop(cola)
        g = ganancia(c)                       # recalcular: la cota guardada puede ser vieja
        if g <= 0:
            # las cotas son superiores y solo decrecen: si el tope no gana nada, nadie gana
            break
        if cola and g < -cola[0][0]:          # otra podría ganar más: reinsertar y seguir
            heapq.heappush(cola, (-g, c))
            continue
        elegidas.append(c)
        actual.update(dijkstra_podado(inv, c, actual))
    return elegidas


# ---------------------------------------------------------------- p-centro
def p_centro(
    grafo: Grafo, dist: dict[int, float], pesos: dict[int, float], k: int, invertido: Grafo | None = None
) -> list[int]:
    """Heurística del punto más lejano para p-centro.

    Cada paso pone una estación EN el nodo que hoy está más lejos de todas. Es la heurística
    clásica (Gonzalez) y es muy sensible a los puntos extremos: la ciudad entera se mide por
    su peor rincón.

    Solo considera nodos con peso > 0 como candidatos y como referencia: con datos de demanda,
    los lugares sin demanda no deben decidir la distancia máxima. Si todos los nodos activos
    ya están a distancia 0, devuelve menos de `k` (no hay dónde poner más).
    """
    inv = invertido if invertido is not None else grafo.invertido()
    actual = dict(dist)
    activos = [n for n in actual if pesos.get(n, 0.0) > 0]
    if k <= 0 or not activos:
        return []

    elegidas: list[int] = []
    for _ in range(k):
        f = max(activos, key=lambda n: (actual[n], -n))     # desempate determinista
        if actual[f] <= 0:
            break
        elegidas.append(f)
        actual.update(dijkstra_podado(inv, f, actual))
    return elegidas


# ------------------------------------------------------------- comparación
def comparar_objetivos(
    grafo: Grafo,
    nodos_estacion: set[int],
    pesos: dict[int, float],
    k: int,
    radio: float,
    invertido: Grafo | None = None,
) -> list[Resultado]:
    """Corre los tres criterios con el mismo `k` y los evalúa igual. El primero es 'Actual'."""
    inv = invertido if invertido is not None else grafo.invertido()
    dist, _ = distancias_a_estacion(grafo, nodos_estacion, inv)

    elegidas = {
        COBERTURA: [c.nodo for c in proponer_estaciones(grafo, dist, pesos, k, radio, inv)],
        MEDIANA: p_mediana(grafo, dist, pesos, k, inv),
        CENTRO: p_centro(grafo, dist, pesos, k, inv),
    }
    resultados = [Resultado(ACTUAL, NOMBRES[ACTUAL], [], evaluar(grafo, inv, nodos_estacion, pesos, radio))]
    for clave, nodos in elegidas.items():
        resultados.append(Resultado(
            clave, NOMBRES[clave], nodos,
            evaluar(grafo, inv, nodos_estacion | set(nodos), pesos, radio),
        ))
    return resultados


def coincidencias(grafo: Grafo, resultados: list[Resultado], radio: float) -> dict[tuple[str, str], int]:
    """Cuántas ubicaciones del criterio A tienen una del criterio B a <= `radio` metros (línea recta)."""
    nuevos = [r for r in resultados if r.clave != ACTUAL]
    cerca: dict[tuple[str, str], int] = {}
    for a in nuevos:
        for b in nuevos:
            if a.clave == b.clave:
                continue
            cerca[(a.clave, b.clave)] = sum(
                1 for na in a.nodos
                if any(haversine(*grafo.coords(na), *grafo.coords(nb)) <= radio for nb in b.nodos)
            )
    return cerca