"""Valida los criterios de ubicación (p-mediana, p-centro) y su comparación (corre sin conexión)."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import analisis
import comparacion
import datos
from dijkstra import dijkstra, dijkstra_multifuente, dijkstra_podado
from grafo import Grafo


def grafo_aleatorio(n=120, m=450, semilla=11) -> Grafo:
    rnd = random.Random(semilla)
    g = Grafo()
    for i in range(n):
        g.agregar_nodo(i, 0.0, 0.0)
    for _ in range(m):
        g.agregar_arista(rnd.randrange(n), rnd.randrange(n), rnd.uniform(20, 200))
    return g


class TestPodado(unittest.TestCase):
    def test_vs_dijkstra_completo(self):
        g = grafo_aleatorio()
        g = g.subgrafo(g.componente_fuerte_mayor())
        inv = g.invertido()
        actual, _ = dijkstra_multifuente(inv, [g.nodos()[0], g.nodos()[7]])
        rnd = random.Random(1)
        for c in rnd.sample(g.nodos(), 30):
            completo, _, _ = dijkstra(inv, c)
            esperado = {n: d for n, d in completo.items() if d < actual[n]}
            esperado = esperado if actual[c] > 0 else {}
            if actual[c] > 0:
                esperado[c] = 0.0
            obtenido = dijkstra_podado(inv, c, actual)
            self.assertEqual(set(obtenido), set(esperado))
            for n, d in esperado.items():
                self.assertAlmostEqual(obtenido[n], d, places=6)

    def test_una_fuente_actual_no_mejora(self):
        g = grafo_aleatorio()
        inv = g.invertido()
        actual, _ = dijkstra_multifuente(inv, [3])
        self.assertEqual(dijkstra_podado(inv, 3, actual), {})


class Base(unittest.TestCase):
    def setUp(self):
        g, estaciones = datos.ciudad_sintetica(filas=20, columnas=20)
        self.g = g.subgrafo(g.componente_fuerte_mayor())
        por_nodo, _ = datos.ubicar_estaciones(self.g, estaciones)
        self.base = set(por_nodo)
        self.inv = self.g.invertido()
        self.pesos = analisis.pesos_por_longitud(self.g)
        self.radio = 250.0
        self.dist, _ = analisis.distancias_a_estacion(self.g, self.base, self.inv)

    def distancias(self, nodos):
        d, _ = analisis.distancias_a_estacion(self.g, self.base | set(nodos), self.inv)
        return d

    def suma_ponderada(self, nodos):
        return sum(self.pesos[n] * d for n, d in self.distancias(nodos).items())


class TestPMediana(Base):
    def test_perezoso_igual_al_ingenuo(self):
        """El greedy perezoso tiene que dar el mismo objetivo que reevaluar todo en cada paso."""
        k = 4
        elegidas = comparacion.p_mediana(self.g, self.dist, self.pesos, k, self.inv)
        self.assertEqual(len(elegidas), k)

        ingenuas: list[int] = []
        for i in range(k):
            candidatos = [n for n in self.g.nodos() if n not in self.base and n not in ingenuas]
            mejor = min(candidatos, key=lambda c: (self.suma_ponderada(ingenuas + [c]), c))
            ingenuas.append(mejor)
            self.assertAlmostEqual(
                self.suma_ponderada(elegidas[: i + 1]), self.suma_ponderada(ingenuas), places=6,
                msg=f"paso {i + 1}",
            )

    def test_el_objetivo_baja_en_cada_paso(self):
        elegidas = comparacion.p_mediana(self.g, self.dist, self.pesos, 8, self.inv)
        valores = [self.suma_ponderada(elegidas[:i]) for i in range(len(elegidas) + 1)]
        self.assertTrue(all(b < a for a, b in zip(valores, valores[1:])))

    def test_no_repite_ni_elige_estaciones_existentes(self):
        elegidas = comparacion.p_mediana(self.g, self.dist, self.pesos, 10, self.inv)
        self.assertEqual(len(set(elegidas)), len(elegidas))
        self.assertFalse(set(elegidas) & self.base)


class TestPCentro(Base):
    def test_primera_es_la_mas_lejana(self):
        elegidas = comparacion.p_centro(self.g, self.dist, self.pesos, 1, self.inv)
        self.assertEqual(self.dist[elegidas[0]], max(self.dist.values()))

    def test_maximo_no_crece(self):
        elegidas = comparacion.p_centro(self.g, self.dist, self.pesos, 6, self.inv)
        maximos = [max(self.distancias(elegidas[:i]).values()) for i in range(len(elegidas) + 1)]
        self.assertTrue(all(b <= a for a, b in zip(maximos, maximos[1:])))
        self.assertLess(maximos[-1], maximos[0])

    def test_ignora_nodos_sin_peso(self):
        """Con demanda, un lugar sin demanda (peso 0) no puede decidir dónde va la estación."""
        lejano = max(self.dist, key=self.dist.get)
        pesos = {n: (0.0 if n == lejano else 1.0) for n in self.g.nodos()}
        elegida = comparacion.p_centro(self.g, self.dist, pesos, 1, self.inv)[0]
        self.assertNotEqual(elegida, lejano)
        activos = [n for n in self.dist if pesos[n] > 0]
        self.assertEqual(self.dist[elegida], max(self.dist[n] for n in activos))


class TestComparacion(Base):
    def test_evaluar_vs_calculo_manual(self):
        m = comparacion.evaluar(self.g, self.inv, self.base, self.pesos, self.radio)
        total = sum(self.pesos[n] for n in self.dist)
        self.assertAlmostEqual(m["media_m"], sum(self.pesos[n] * d for n, d in self.dist.items()) / total, 6)
        self.assertAlmostEqual(m["max_m"], max(self.dist.values()), 6)
        self.assertAlmostEqual(
            m["cobertura_pct"],
            100 * sum(self.pesos[n] for n, d in self.dist.items() if d <= self.radio) / total, 6,
        )

    def test_cada_criterio_mejora_su_propia_metrica(self):
        r = {x.clave: x for x in comparacion.comparar_objetivos(
            self.g, self.base, self.pesos, 5, self.radio, self.inv)}
        self.assertEqual(list(r), ["actual", "cobertura", "p_mediana", "p_centro"])
        a = r["actual"].metricas
        self.assertGreater(r["cobertura"].metricas["cobertura_pct"], a["cobertura_pct"])
        self.assertLess(r["p_mediana"].metricas["media_m"], a["media_m"])
        self.assertLess(r["p_centro"].metricas["max_m"], a["max_m"])

    def test_p_mediana_gana_en_media_y_p_centro_en_maximo(self):
        """Cada criterio debería ser el mejor en la métrica que optimiza (o quedar muy cerca).

        Son heurísticas greedy: no garantizan ganarle a otro criterio por poco, así que se admite 1 %.
        """
        r = {x.clave: x.metricas for x in comparacion.comparar_objetivos(
            self.g, self.base, self.pesos, 5, self.radio, self.inv)}
        self.assertLessEqual(r["p_mediana"]["media_m"], r["cobertura"]["media_m"] * 1.01)
        self.assertLessEqual(r["p_centro"]["max_m"], r["p_mediana"]["max_m"] + 1e-6)
        self.assertLessEqual(r["p_centro"]["max_m"], r["cobertura"]["max_m"] + 1e-6)

    def test_ubicaciones_nuevas_no_son_estaciones(self):
        for x in comparacion.comparar_objetivos(self.g, self.base, self.pesos, 6, self.radio, self.inv):
            self.assertFalse(set(x.nodos) & self.base)
            self.assertEqual(len(set(x.nodos)), len(x.nodos))

    def test_coincidencias(self):
        res = comparacion.comparar_objetivos(self.g, self.base, self.pesos, 4, self.radio, self.inv)
        co = comparacion.coincidencias(self.g, res, 1.0)       # radio ínfimo: solo el mismo nodo
        iguales = lambda a, b: len(set(next(r for r in res if r.clave == a).nodos) &
                                   set(next(r for r in res if r.clave == b).nodos))
        self.assertEqual(co[("p_mediana", "p_centro")], iguales("p_mediana", "p_centro"))
        self.assertEqual(co[("p_centro", "p_mediana")], iguales("p_mediana", "p_centro"))
        # con un radio enorme todas las ubicaciones tienen alguna del otro criterio cerca
        todo = comparacion.coincidencias(self.g, res, 1e9)
        for (a, _), n in todo.items():
            self.assertEqual(n, len(next(r for r in res if r.clave == a).nodos))
        # y un radio mayor nunca da menos coincidencias que uno menor
        medio = comparacion.coincidencias(self.g, res, 600.0)
        for par in medio:
            self.assertGreaterEqual(medio[par], co[par])


if __name__ == "__main__":
    unittest.main()
