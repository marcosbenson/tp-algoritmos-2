"""Valida el análisis de cobertura (corre sin conexión)."""
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

import analisis
import datos
from dijkstra import dijkstra, dijkstra_multifuente
from grafo import Estacion, Grafo


def a_networkx(g: Grafo) -> nx.DiGraph:
    nxg = nx.DiGraph()
    nxg.add_nodes_from(g.nodos())
    for u in g.nodos():
        for v, w in g.vecinos(u):
            nxg.add_edge(u, v, weight=w)
    return nxg


def grafo_aleatorio(n=150, m=500, semilla=3) -> Grafo:
    rnd = random.Random(semilla)
    g = Grafo()
    for i in range(n):
        g.agregar_nodo(i, 0.0, 0.0)
    for _ in range(m):
        g.agregar_arista(rnd.randrange(n), rnd.randrange(n), rnd.uniform(20, 200))
    return g


class TestGrafo(unittest.TestCase):
    def test_invertido(self):
        g = grafo_aleatorio()
        inv = g.invertido()
        self.assertEqual(inv.cantidad_aristas(), g.cantidad_aristas())
        for u in g.nodos():
            for v, w in g.vecinos(u):
                self.assertEqual(inv.peso(v, u), w)

    def test_componente_fuerte_mayor_vs_networkx(self):
        for semilla in range(5):
            g = grafo_aleatorio(semilla=semilla)
            esperado = max(nx.strongly_connected_components(a_networkx(g)), key=len)
            self.assertEqual(g.componente_fuerte_mayor(), set(esperado))

    def test_subgrafo(self):
        g = grafo_aleatorio()
        comp = g.componente_fuerte_mayor()
        sub = g.subgrafo(comp)
        self.assertEqual(set(sub.nodos()), comp)
        for u in sub.nodos():
            for v, _ in sub.vecinos(u):
                self.assertIn(v, comp)


class TestMultifuente(unittest.TestCase):
    def test_vs_networkx(self):
        g = grafo_aleatorio()
        fuentes = [3, 40, 99]
        dist, cercana = dijkstra_multifuente(g, fuentes)
        esperado = nx.multi_source_dijkstra_path_length(a_networkx(g), fuentes)
        self.assertEqual(set(dist), set(esperado))
        for n, d in esperado.items():
            self.assertAlmostEqual(dist[n], d, places=6)
            # la fuente informada realmente está a esa distancia
            self.assertAlmostEqual(dijkstra(g, cercana[n], {n})[0][n], d, places=6)

    def test_distancia_hacia_la_estacion_usa_el_grafo_invertido(self):
        """nodo -> estación (ir hacia ella) respeta los sentidos de las calles."""
        g = grafo_aleatorio()
        estaciones = [5, 77]
        dist, _ = analisis.distancias_a_estacion(g, estaciones)
        for n in g.nodos():
            ours = dist.get(n, math.inf)
            esperado = min(
                (dijkstra(g, n, {e})[0].get(e, math.inf) for e in estaciones), default=math.inf
            )
            self.assertAlmostEqual(ours, esperado, places=6)

    def test_radio_max(self):
        g = grafo_aleatorio()
        dist, _, _ = dijkstra(g, 0, None, 250.0)
        completo, _, _ = dijkstra(g, 0)
        self.assertTrue(all(d <= 250.0 for d in dist.values()))
        self.assertEqual(set(dist), {n for n, d in completo.items() if d <= 250.0})
        d2, _ = dijkstra_multifuente(g, [0], radio_max=250.0)
        self.assertEqual(d2, dist)


class TestPropuestas(unittest.TestCase):
    def setUp(self):
        g, estaciones = datos.ciudad_sintetica(filas=30, columnas=30)
        self.g = g.subgrafo(g.componente_fuerte_mayor())
        por_nodo, _ = datos.ubicar_estaciones(self.g, estaciones)
        self.nodos_est = set(por_nodo)
        self.inv = self.g.invertido()
        self.dist, _ = analisis.distancias_a_estacion(self.g, self.nodos_est, self.inv)
        self.pesos = analisis.pesos_por_longitud(self.g)
        self.radio = 200.0     # a 400 m esta ciudad de prueba queda casi toda cubierta

    def test_primera_elegida_es_la_mejor_por_fuerza_bruta(self):
        cand = analisis.proponer_estaciones(self.g, self.dist, self.pesos, 1, self.radio, self.inv)
        self.assertEqual(len(cand), 1)
        descubiertos = {n for n, d in self.dist.items() if d > self.radio}
        mejor = 0.0
        for c in descubiertos:
            alcance, _, _ = dijkstra(self.inv, c, None, self.radio)
            mejor = max(mejor, sum(self.pesos[n] for n in alcance if n in descubiertos))
        self.assertAlmostEqual(cand[0].ganancia, mejor, places=6)

    def test_cobertura_acumulada_coincide_con_recalcular(self):
        """Agregar las estaciones propuestas y recalcular da exactamente el % que se informó."""
        cand = analisis.proponer_estaciones(self.g, self.dist, self.pesos, 8, self.radio, self.inv)
        self.assertEqual(len(cand), 8)
        nuevas, _ = analisis.distancias_a_estacion(
            self.g, self.nodos_est | {c.nodo for c in cand}, self.inv
        )
        real = analisis.porcentaje_cubierto(nuevas, self.pesos, self.radio)
        self.assertAlmostEqual(cand[-1].cobertura_acumulada_pct, real, places=6)

    def test_ganancias_no_crecientes_y_cobertura_sube(self):
        cand = analisis.proponer_estaciones(self.g, self.dist, self.pesos, 6, self.radio, self.inv)
        gan = [c.ganancia for c in cand]
        self.assertEqual(gan, sorted(gan, reverse=True))
        acum = [c.cobertura_acumulada_pct for c in cand]
        self.assertEqual(acum, sorted(acum))

    def test_candidatas_estan_en_zonas_descubiertas(self):
        cand = analisis.proponer_estaciones(self.g, self.dist, self.pesos, 5, self.radio, self.inv)
        for c in cand:
            self.assertGreater(self.dist[c.nodo], self.radio)
            self.assertNotIn(c.nodo, self.nodos_est)

    def test_perezoso_igual_al_ingenuo_con_muchos_pasos(self):
        """Regresión: un candidato con ganancia real 0 en el tope de la cola no debe cortar la búsqueda."""
        k = 25
        cand = analisis.proponer_estaciones(self.g, self.dist, self.pesos, k, self.radio, self.inv)
        descubiertos = {n for n, d in self.dist.items() if d > self.radio}
        zonas = {c: {n for n in dijkstra(self.inv, c, None, self.radio)[0] if n in descubiertos}
                 for c in descubiertos}
        sin_cubrir = set(descubiertos)
        for i in range(k):
            mejor = max(sum(self.pesos[n] for n in z & sin_cubrir) for z in zonas.values())
            if mejor <= 0:
                self.assertEqual(len(cand), i)      # solo puede parar si ya no queda nada por cubrir
                break
            self.assertGreater(len(cand), i, msg=f"paró en {len(cand)} con ganancia posible {mejor}")
            self.assertAlmostEqual(cand[i].ganancia, mejor, places=6, msg=f"paso {i + 1}")
            sin_cubrir -= zonas[cand[i].nodo]

    def test_sin_descubiertos_no_propone(self):
        cand = analisis.proponer_estaciones(self.g, self.dist, self.pesos, 5, 1e9, self.inv)
        self.assertEqual(cand, [])


class TestQuitar(unittest.TestCase):
    def setUp(self):
        g, estaciones = datos.ciudad_sintetica(filas=30, columnas=30)
        rnd = random.Random(5)                       # estaciones extra para que haya solapamiento
        extra = [Estacion(f"x{i}", f"extra {i}", *g.coords(rnd.randrange(900))) for i in range(40)]
        self.g = g.subgrafo(g.componente_fuerte_mayor())
        self.por_nodo, _ = datos.ubicar_estaciones(self.g, estaciones + extra)
        self.inv = self.g.invertido()
        self.pesos = analisis.pesos_por_longitud(self.g)
        self.radio = 300.0

    def cobertura_sin(self, por_nodo, excluidas):
        """Recalcula desde cero (Dijkstra multi-fuente) sin las estaciones `excluidas`."""
        nodos = {n for n, ests in por_nodo.items() if any(e.id not in excluidas for e in ests)}
        dist, _ = analisis.distancias_a_estacion(self.g, nodos, self.inv)
        return analisis.porcentaje_cubierto(dist, self.pesos, self.radio)

    def test_primera_es_la_de_menor_perdida(self):
        base = self.cobertura_sin(self.por_nodo, set())
        mejor = min(
            base - self.cobertura_sin(self.por_nodo, {e.id})
            for ests in self.por_nodo.values() for e in ests
        )
        q = analisis.proponer_quitar(self.g, self.por_nodo, self.pesos, 1, self.radio, self.inv)
        self.assertAlmostEqual(q[0].perdida_pct, mejor, places=6)

    def test_cobertura_restante_coincide_con_recalcular_en_cada_paso(self):
        q = analisis.proponer_quitar(self.g, self.por_nodo, self.pesos, 15, self.radio, self.inv)
        self.assertEqual(len(q), 15)
        for i in range(len(q)):
            excluidas = {x.estacion.id for x in q[: i + 1]}
            self.assertAlmostEqual(
                q[i].cobertura_restante_pct, self.cobertura_sin(self.por_nodo, excluidas), places=6
            )

    def test_perdidas_no_decrecientes(self):
        q = analisis.proponer_quitar(self.g, self.por_nodo, self.pesos, 25, self.radio, self.inv)
        perdidas = [x.perdida for x in q]
        self.assertEqual(perdidas, sorted(perdidas))

    def test_no_modifica_la_entrada(self):
        antes = {n: [e.id for e in ests] for n, ests in self.por_nodo.items()}
        analisis.proponer_quitar(self.g, self.por_nodo, self.pesos, 10, self.radio, self.inv)
        self.assertEqual({n: [e.id for e in ests] for n, ests in self.por_nodo.items()}, antes)

    def test_dos_estaciones_en_el_mismo_nodo_una_sobra(self):
        por_nodo = {n: list(v) for n, v in self.por_nodo.items()}
        nodo = next(iter(por_nodo))
        por_nodo[nodo].append(Estacion("dup", "duplicada", *self.g.coords(nodo)))
        base = self.cobertura_sin(por_nodo, set())
        q = analisis.proponer_quitar(self.g, por_nodo, self.pesos, 1, self.radio, self.inv)
        self.assertEqual(q[0].perdida, 0.0)
        self.assertEqual(q[0].compartida_pct, 100.0)
        self.assertAlmostEqual(self.cobertura_sin(por_nodo, {q[0].estacion.id}), base, places=9)

    def test_nunca_quita_la_ultima(self):
        total = sum(len(v) for v in self.por_nodo.values())
        q = analisis.proponer_quitar(self.g, self.por_nodo, self.pesos, 10**6, self.radio, self.inv)
        self.assertEqual(len(q), total - 1)


class TestEspaciadoYDemanda(unittest.TestCase):
    def test_espaciado_vs_networkx(self):
        g, estaciones = datos.ciudad_sintetica(filas=25, columnas=25)
        g = g.subgrafo(g.componente_fuerte_mayor())
        por_nodo, _ = datos.ubicar_estaciones(g, estaciones)
        esp = analisis.espaciado_entre_estaciones(g, por_nodo)
        nxg = a_networkx(g)
        for nodo, ests in por_nodo.items():
            otros = [n for n in por_nodo if n != nodo]
            esperado = min(
                (nx.dijkstra_path_length(nxg, nodo, o) for o in otros if nx.has_path(nxg, nodo, o)),
                default=math.inf,
            )
            if len(ests) > 1:          # varias estaciones en el mismo nodo: distancia 0 por diseño
                esperado = 0.0
            for e in ests:
                self.assertAlmostEqual(esp[e.id], esperado, places=6)

    def test_demanda_desde_csv(self):
        g, _ = datos.ciudad_sintetica(filas=10, columnas=10)
        lat, lon = g.coords(55)
        with tempfile.TemporaryDirectory() as t:
            ruta = Path(t) / "d.csv"
            ruta.write_text(
                "lat,lon,peso\n"
                f"{lat},{lon},5\n"
                f"{lat},{lon},2.5\n"          # mismo nodo: se suman
                "19.43,-99.15,9\n",           # lejos de la red: se ignora
                encoding="utf-8",
            )
            demanda, ignorados = analisis.cargar_demanda(ruta, g)
        self.assertEqual(demanda, {55: 7.5})
        self.assertEqual(ignorados, 1)

    def test_percentil_y_cobertura(self):
        dist = {1: 100.0, 2: 300.0, 3: 500.0, 4: 900.0}
        pesos = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self.assertEqual(analisis.porcentaje_cubierto(dist, pesos, 300.0), 50.0)
        self.assertEqual(analisis.percentil_ponderado(dist, pesos, 0.5), 300.0)
        self.assertEqual(analisis.percentil_ponderado(dist, pesos, 1.0), 900.0)


if __name__ == "__main__":
    unittest.main()
