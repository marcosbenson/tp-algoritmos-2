"""Compara tres criterios para ubicar estaciones nuevas: máxima cobertura, p-mediana y p-centro.

Uso:
    python comparar.py --demo                       # ciudad sintética, sin conexión
    python comparar.py                              # datos reales ya descargados (ver main.py --descargar)
    python comparar.py --k 20 --radio 400           # estaciones nuevas por criterio y radio de cobertura
    python comparar.py --demanda puntos.csv         # ponderar por demanda (CSV con lat,lon[,peso])

Genera en `resultados/`: comparacion.png, comparacion_ubicaciones.csv y comparacion_metricas.csv.
"""
from __future__ import annotations

import argparse
import csv
import time

import comparacion
import grafico
from preparacion import agregar_argumentos_comunes, preparar

COLUMNAS = (
    ("cobertura_pct", "cobertura", True),
    ("media_m", "dist. media", False),
    ("p90_m", "p90", False),
    ("max_m", "máxima", False),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    agregar_argumentos_comunes(ap)
    ap.add_argument("--k", type=int, default=20, help="estaciones nuevas por criterio")
    args = ap.parse_args()

    ctx = preparar(args)
    grafo, nodos_estacion = ctx.grafo, set(ctx.por_nodo)

    t0 = time.perf_counter()
    resultados = comparacion.comparar_objetivos(grafo, nodos_estacion, ctx.pesos, args.k, args.radio, ctx.invertido)
    print(f"\n== Comparación de criterios: {args.k} estaciones nuevas, radio {args.radio:.0f} m "
          f"({time.perf_counter() - t0:.1f} s) ==")
    print(f"  {'Criterio':<18}{'nuevas':>7}{'cobertura':>12}{'dist. media':>13}{'p90':>10}{'máxima':>10}")
    for r in resultados:
        m = r.metricas
        print(f"  {r.nombre:<18}{len(r.nodos) or '—':>7}{m['cobertura_pct']:>10.1f} %"
              f"{m['media_m']:>11,.0f} m{m['p90_m']:>8,.0f} m{m['max_m']:>8,.0f} m")

    nuevos = [r for r in resultados if r.clave != comparacion.ACTUAL]

    def mejores(clave: str, mayor: bool) -> str:
        valores = [r.metricas[clave] for r in nuevos]
        mejor = max(valores) if mayor else min(valores)
        return " = ".join(r.nombre for r in nuevos if abs(r.metricas[clave] - mejor) <= 0.005 * abs(mejor))

    print("\n  Mejor en cada métrica (se considera empate si difieren menos de 0,5 %):\n    "
          + " · ".join(f"{etiqueta} → {mejores(clave, mayor)}" for clave, etiqueta, mayor in COLUMNAS))

    solape = 2 * args.radio          # dos estaciones solapan su zona de cobertura a menos de 2 × radio
    co = comparacion.coincidencias(grafo, resultados, solape)
    print(f"\n== ¿Eligen los mismos lugares? Ubicaciones de la fila con otra de la columna a ≤ {solape:.0f} m "
          f"(2 × radio: ahí las zonas de cobertura se solapan) ==")
    print(f"  {'':<18}" + "".join(f"{r.nombre:>18}" for r in nuevos))
    for a in nuevos:
        print(f"  {a.nombre:<18}" + "".join(
            f"{'—' if a.clave == b.clave else f'{co[(a.clave, b.clave)]} de {len(a.nodos)}':>18}" for b in nuevos))
    print("\n  Lectura: p-centro mide la ciudad por su peor rincón (sensible a lugares extremos o "
          "sin demanda real);\n  p-mediana baja la distancia media y tiende a reforzar zonas densas; "
          "máxima cobertura lleva\n  a la mayor cantidad de gente a un radio fijo, sin importar cuánto "
          "más lejos quedan los demás.")

    args.salida.mkdir(parents=True, exist_ok=True)
    with open(args.salida / "comparacion_ubicaciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterio", "orden", "lat", "lon", "google_maps"])
        for r in nuevos:
            for i, n in enumerate(r.nodos, 1):
                lat, lon = grafo.coords(n)
                w.writerow([r.nombre, i, f"{lat:.6f}", f"{lon:.6f}",
                            f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"])
    with open(args.salida / "comparacion_metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterio", "nuevas", "cobertura_pct", "distancia_media_m", "p90_m", "maxima_m"])
        for r in resultados:
            m = r.metricas
            w.writerow([r.nombre, len(r.nodos), f"{m['cobertura_pct']:.2f}", f"{m['media_m']:.0f}",
                        f"{m['p90_m']:.0f}", f"{m['max_m']:.0f}"])
    grafico.dibujar_comparacion(
        grafo, nodos_estacion, resultados, args.radio, args.salida / "comparacion.png",
        f"Tres criterios para ubicar {args.k} estaciones nuevas · radio {args.radio:.0f} m",
    )
    print(f"\nArchivos en {args.salida}/: comparacion.png, comparacion_ubicaciones.csv, comparacion_metricas.csv")


if __name__ == "__main__":
    main()
