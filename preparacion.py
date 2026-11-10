"""Carga y preparación de datos compartida por `analizar.py` y `comparar.py`."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import analisis
import datos
from grafo import Estacion, Grafo
from main import cargar_datos


@dataclass
class Contexto:
    grafo: Grafo                              # solo la mayor componente fuertemente conexa
    por_nodo: dict[int, list[Estacion]]       # nodo -> estaciones ubicadas ahí
    pesos: dict[int, float]                   # peso de cada nodo (demanda o metros de calle)
    invertido: Grafo


def agregar_argumentos_comunes(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--demo", action="store_true", help="usar ciudad sintética")
    ap.add_argument("--descargar", action="store_true", help="forzar descarga de datos reales")
    ap.add_argument("--radio", type=float, default=400.0, help="metros por calle para 'cubierto'")
    ap.add_argument("--demanda", type=Path, help="CSV lat,lon[,peso] para ponderar la cobertura")
    ap.add_argument("--salida", type=Path, default=Path("resultados"), help="carpeta de salida")


def preparar(args: argparse.Namespace) -> Contexto:
    grafo_completo, estaciones = cargar_datos(args)
    comp = grafo_completo.componente_fuerte_mayor()
    grafo = grafo_completo.subgrafo(comp) if len(comp) < grafo_completo.cantidad_nodos() else grafo_completo
    quitados = grafo_completo.cantidad_nodos() - grafo.cantidad_nodos()
    print(f"Red vial: {grafo.cantidad_nodos():,} nodos, {grafo.cantidad_aristas():,} aristas dirigidas"
          + (f" ({quitados:,} nodos descartados por estar aislados o sin salida)" if quitados else ""))

    por_nodo, descartadas = datos.ubicar_estaciones(grafo, estaciones)
    usadas = len(estaciones) - len(descartadas)
    print(f"Estaciones: {usadas} de {len(estaciones)} ubicadas en {len(por_nodo)} intersecciones")
    if usadas < len(estaciones) / 2:
        raise SystemExit("Más de la mitad de las estaciones están fuera de la red: ¿son de otra ciudad?")

    if args.demanda:
        demanda, ignorados = analisis.cargar_demanda(args.demanda, grafo)
        if not demanda:
            raise SystemExit("Ningún punto del CSV de demanda cae sobre la red vial.")
        pesos = demanda
        print(f"Ponderando por demanda: {len(demanda):,} nodos con peso ({ignorados} puntos fuera de la red)")
    else:
        pesos = analisis.pesos_por_longitud(grafo)
        print("Ponderando por metros de calle (sin datos de demanda)")

    return Contexto(grafo, por_nodo, pesos, grafo.invertido())
