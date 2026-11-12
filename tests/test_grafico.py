"""Prueba de humo de los tres gráficos: se generan sin error y el archivo es una imagen PNG."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import analisis
import comparacion
import datos
import grafico
from dijkstra import camino_minimo, dijkstra
from grafo import haversine

PNG = b"\x89PNG\r\n\x1a\n"


class TestGraficos(unittest.TestCase):
    def setUp(self):
        g, estaciones = datos.ciudad_sintetica(filas=20, columnas=20)
        self.g = g.subgrafo(g.componente_fuerte_mayor())
        self.por_nodo, _ = datos.ubicar_estaciones(self.g, estaciones)
        self.inv = self.g.invertido()
        self.pesos = analisis.pesos_por_longitud(self.g)
        self.radio = 250.0
        self.tmp = Path(tempfile.mkdtemp())

    def verificar(self, ruta: Path):
        self.assertTrue(ruta.exists())
        self.assertEqual(ruta.read_bytes()[:8], PNG)

    def test_cobertura_con_y_sin_propuestas(self):
        dist, _ = analisis.distancias_a_estacion(self.g, set(self.por_nodo), self.inv)
        cand = analisis.proponer_estaciones(self.g, dist, self.pesos, 4, self.radio, self.inv)
        quit_ = analisis.proponer_quitar(self.g, self.por_nodo, self.pesos, 3, self.radio, self.inv)
        grafico.dibujar_cobertura(self.g, dist, self.pesos, set(self.por_nodo), cand, self.radio,
                                  self.tmp / "a.png", "t", quit_, self.inv)
        self.verificar(self.tmp / "a.png")
        grafico.dibujar_cobertura(self.g, dist, self.pesos, set(self.por_nodo), [], self.radio,
                                  self.tmp / "b.png", "t")          # sin propuestas ni redundantes
        self.verificar(self.tmp / "b.png")

    def test_comparacion(self):
        res = comparacion.comparar_objetivos(self.g, set(self.por_nodo), self.pesos, 4, self.radio, self.inv)
        grafico.dibujar_comparacion(self.g, set(self.por_nodo), res, self.radio, self.tmp / "c.png", "t", self.inv)
        self.verificar(self.tmp / "c.png")

    def test_ruta_con_y_sin_camino(self):
        nodos = self.g.nodos()
        o, d = nodos[0], nodos[-1]
        metros, camino = camino_minimo(self.g, o, d)
        explorados, _, _ = dijkstra(self.g, o, [d])
        e = next(iter(self.por_nodo.values()))[0]
        po, pd = self.g.coords(o), self.g.coords(d)
        recta = haversine(*po, *pd)
        grafico.dibujar_ruta(self.g, explorados, camino, self.por_nodo, po, pd, (e, 100.0), (e, 90.0),
                             metros, recta, self.tmp / "d.png", "t")
        self.verificar(self.tmp / "d.png")
        grafico.dibujar_ruta(self.g, explorados, [], self.por_nodo, po, pd, None, (e, 90.0),
                             None, recta, self.tmp / "e.png", "t")   # sin camino ni estación por calle
        self.verificar(self.tmp / "e.png")


if __name__ == "__main__":
    unittest.main()
