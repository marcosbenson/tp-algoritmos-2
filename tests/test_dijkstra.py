"""Valida nuestro Dijkstra contra networkx (corre sin conexión)."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

import datos
from dijkstra import camino_minimo, dijkstra
from grafo import Grafo


def a_networkx(g: Grafo) -> nx.DiGraph:
    nxg = nx.DiGraph()
    nxg.add_nodes_from(g.nodos())
    for u in g.nodos():
        for v, w in g.vecinos(u):
            nxg.add_edge(u, v, weight=w)
    return nxg


class TestDijkstra(unittest.TestCase):
    def test_grafo_aleatorio_dirigido(self):
        rnd = random.Random(1)
        g = Grafo()
        for n in range(200):
            g.agregar_nodo(n, 0.0, 0.0)
        for _ in range(800):
            g.agregar_arista(rnd.randrange(200), rnd.randrange(200), rnd.uniform(1, 100))
        nxg = a_networkx(g)
        for _ in range(100):
            o, d = rnd.randrange(200), rnd.randrange(200)
            ours, camino = camino_minimo(g, o, d)
            if nx.has_path(nxg, o, d):
                self.assertAlmostEqual(ours, nx.dijkstra_path_length(nxg, o, d), places=6)
                suma = sum(g.peso(a, b) for a, b in zip(camino, camino[1:]))
                self.assertAlmostEqual(suma, ours, places=6)
            else:
                self.assertEqual((ours, camino), (math.inf, []))

    def test_ciudad_sintetica(self):
        g, _ = datos.ciudad_sintetica()
        nxg = a_networkx(g)
        rnd = random.Random(2)
        nodos = g.nodos()
        for _ in range(40):
            o, d = rnd.choice(nodos), rnd.choice(nodos)
            ours, _ = camino_minimo(g, o, d)
            esperado = nx.dijkstra_path_length(nxg, o, d) if nx.has_path(nxg, o, d) else math.inf
            self.assertAlmostEqual(ours, esperado, places=6)

    def test_distancias_completas(self):
        g, _ = datos.ciudad_sintetica(filas=20, columnas=20)
        dist, _, _ = dijkstra(g, 0)
        esperado = nx.single_source_dijkstra_path_length(a_networkx(g), 0)
        self.assertEqual(set(dist), set(esperado))
        for n, d in esperado.items():
            self.assertAlmostEqual(dist[n], d, places=6)

    def test_origen_igual_destino(self):
        g, _ = datos.ciudad_sintetica(filas=5, columnas=5)
        self.assertEqual(camino_minimo(g, 3, 3), (0.0, [3]))

    def test_rechaza_peso_negativo(self):
        g = Grafo()
        g.agregar_nodo(0, 0, 0)
        g.agregar_nodo(1, 0, 0)
        with self.assertRaises(ValueError):
            g.agregar_arista(0, 1, -1)


if __name__ == "__main__":
    unittest.main()
