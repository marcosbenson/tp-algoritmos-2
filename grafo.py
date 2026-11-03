"""Grafo dirigido y pesado (lista de adyacencia) para la red vial."""
from __future__ import annotations

import math
from collections.abc import ItemsView
from dataclasses import dataclass

import numpy as np

RADIO_TIERRA_M = 6_371_000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos (línea recta sobre la esfera)."""
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df = f2 - f1
    dl = math.radians(lon2 - lon1)
    a = math.sin(df / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin(dl / 2) ** 2
    return 2 * RADIO_TIERRA_M * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Estacion:
    id: str
    nombre: str
    lat: float
    lon: float


class Grafo:
    """Nodos = intersecciones; aristas dirigidas = tramos de calle (peso en metros)."""

    def __init__(self) -> None:
        self._ady: dict[int, dict[int, float]] = {}
        self._coords: dict[int, tuple[float, float]] = {}
        self._indice: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # --- construcción ---
    def agregar_nodo(self, nodo: int, lat: float, lon: float) -> None:
        self._ady.setdefault(nodo, {})
        self._coords[nodo] = (lat, lon)
        self._indice = None

    def agregar_arista(self, u: int, v: int, peso: float) -> None:
        """Agrega u -> v. Si ya existe (calles paralelas), conserva el menor peso."""
        if u not in self._ady or v not in self._ady:
            raise KeyError(f"Arista {u}->{v}: nodo inexistente")
        if peso < 0:
            raise ValueError("Dijkstra requiere pesos no negativos")
        actual = self._ady[u].get(v)
        if actual is None or peso < actual:
            self._ady[u][v] = peso

    # --- consulta ---
    def vecinos(self, u: int) -> ItemsView[int, float]:
        return self._ady[u].items()

    def nodos(self) -> list[int]:
        return list(self._ady)

    def coords(self, nodo: int) -> tuple[float, float]:
        return self._coords[nodo]

    def cantidad_nodos(self) -> int:
        return len(self._ady)

    def cantidad_aristas(self) -> int:
        return sum(len(v) for v in self._ady.values())

    def peso(self, u: int, v: int) -> float:
        return self._ady[u][v]

    def nodo_mas_cercano(self, lat: float, lon: float) -> int:
        """Nodo más cercano en línea recta (aprox. equirectangular, suficiente a escala ciudad)."""
        if self._indice is None:
            ids = np.fromiter(self._coords.keys(), dtype=np.int64, count=len(self._coords))
            lats = np.array([c[0] for c in self._coords.values()])
            lons = np.array([c[1] for c in self._coords.values()])
            self._indice = (ids, lats, lons)
        ids, lats, lons = self._indice
        dx = (lons - lon) * math.cos(math.radians(lat))
        dy = lats - lat
        return int(ids[np.argmin(dx * dx + dy * dy)])

    # --- persistencia ---
    def a_dict(self) -> dict:
        return {
            "nodos": [[n, lat, lon] for n, (lat, lon) in self._coords.items()],
            "aristas": [[u, v, w] for u, vs in self._ady.items() for v, w in vs.items()],
        }

    @classmethod
    def desde_dict(cls, datos: dict) -> Grafo:
        g = cls()
        for n, lat, lon in datos["nodos"]:
            g.agregar_nodo(int(n), lat, lon)
        for u, v, w in datos["aristas"]:
            g.agregar_arista(int(u), int(v), w)
        return g
