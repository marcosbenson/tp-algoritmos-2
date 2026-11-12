"""Análisis de cobertura de estaciones y propuesta de ubicaciones nuevas.

Uso:
    python analizar.py --demo                       # ciudad sintética, sin conexión
    python analizar.py                              # datos reales ya descargados (ver main.py --descargar)
    python analizar.py --radio 400 --k 15           # radio de cobertura y cantidad de estaciones nuevas
    python analizar.py --demanda puntos.csv         # ponderar por demanda (CSV con lat,lon[,peso])

    python analizar.py --quitar 20                  # cuántas estaciones redundantes listar

Genera en `resultados/`: cobertura.png, candidatas.csv, estaciones.csv y redundantes.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import analisis
import grafico
from preparacion import agregar_argumentos_comunes, preparar

UMBRALES_M = (300, 400, 500, 600)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    agregar_argumentos_comunes(ap)
    ap.add_argument("--k", type=int, default=10, help="cantidad de estaciones nuevas a proponer")
    ap.add_argument("--quitar", type=int, default=10,
                    help="cantidad de estaciones redundantes a listar (0 = no calcular)")
    args = ap.parse_args()

    # --- datos -------------------------------------------------------------
    ctx = preparar(args)
    grafo, por_nodo, pesos, inv = ctx.grafo, ctx.por_nodo, ctx.pesos, ctx.invertido

    # --- cobertura actual --------------------------------------------------
    t0 = time.perf_counter()
    dist, _ = analisis.distancias_a_estacion(grafo, set(por_nodo), inv)
    print(f"\n== Cobertura actual (Dijkstra multi-fuente, {time.perf_counter() - t0:.2f} s) ==")
    if not dist:
        print("  No hay estaciones alcanzables en la red: nada que analizar.")
        return
    for u in sorted(set(UMBRALES_M) | {round(args.radio)}):
        marca = "  <- radio elegido" if u == round(args.radio) else ""
        print(f"  a ≤ {u:>3} m de una estación: {analisis.porcentaje_cubierto(dist, pesos, u):5.1f} %{marca}")
    p50, p90, p99 = (analisis.percentil_ponderado(dist, pesos, q) for q in (0.5, 0.9, 0.99))
    print(f"  Distancia a la estación más cercana: mediana {p50:,.0f} m · p90 {p90:,.0f} m · "
          f"p99 {p99:,.0f} m · máx {max(dist.values()):,.0f} m")

    # --- espaciado entre estaciones ---------------------------------------
    esp = analisis.espaciado_entre_estaciones(grafo, por_nodo)
    finitos = sorted(v for v in esp.values() if math.isfinite(v))
    if finitos:
        juntas = sum(1 for v in finitos if v < 150)
        aisladas = sorted(((v, e) for ests in por_nodo.values() for e in ests
                           if math.isfinite(v := esp[e.id])), key=lambda t: -t[0])[:5]
        print("\n== Espaciado entre estaciones (por calle, hasta la vecina más cercana) ==")
        print(f"  mediana {finitos[len(finitos) // 2]:,.0f} m · mín {finitos[0]:,.0f} m · máx {finitos[-1]:,.0f} m")
        print(f"  {juntas} estaciones a menos de 150 m de otra (posible solapamiento)")
        if aisladas:
            print("  Más aisladas: " + "; ".join(f"{e.nombre} ({v:,.0f} m)" for v, e in aisladas))
    else:
        print("\n== Espaciado entre estaciones ==")
        print("  No hay pares de estaciones conectadas por calle.")

    # --- propuestas --------------------------------------------------------
    t0 = time.perf_counter()
    candidatas = analisis.proponer_estaciones(grafo, dist, pesos, args.k, args.radio, inv)
    print(f"\n== {len(candidatas)} ubicaciones propuestas (radio {args.radio:.0f} m, "
          f"{time.perf_counter() - t0:.2f} s) ==")
    if not candidatas:
        print("  No quedan zonas descubiertas con ese radio.")
    else:
        print("   #   latitud    longitud   +cobertura   acumulada   intersecciones")
        for c in candidatas:
            print(f"  {c.orden:>2}  {c.lat:9.5f}  {c.lon:10.5f}   {c.ganancia_pct:6.2f} pp   "
                  f"{c.cobertura_acumulada_pct:6.1f} %   {c.nodos_nuevos:>6}")

    # --- estaciones redundantes --------------------------------------------
    quitadas = []
    if args.quitar > 0:
        t0 = time.perf_counter()
        quitadas = analisis.proponer_quitar(grafo, por_nodo, pesos, args.quitar, args.radio, inv)
        print(f"\n== {len(quitadas)} estaciones que MENOS cobertura aportan (radio {args.radio:.0f} m, "
              f"{time.perf_counter() - t0:.2f} s) ==")
        if quitadas:
            print("   #  estación                              pérdida(m)   pérdida   cobertura   ya cubierto por otras")
            for q in quitadas:
                print(f"  {q.orden:>2}  {q.estacion.nombre[:34]:<34}  {q.perdida:10,.1f}  "
                      f"{q.perdida_pct:6.3f} pp   {q.cobertura_restante_pct:6.1f} %   {q.compartida_pct:6.0f} %")
            gratis = sum(1 for q in quitadas if q.perdida == 0)
            print(f"  {gratis} de {len(quitadas)} se podrían sacar sin perder cobertura a este radio. "
                  "Ojo: la cobertura no mide capacidad ni uso.")
        else:
            print("  No hay estaciones redundantes que sacar (o hay una sola estación).")

    # --- archivos ----------------------------------------------------------
    args.salida.mkdir(parents=True, exist_ok=True)
    with open(args.salida / "redundantes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orden", "id", "nombre", "lat", "lon", "perdida", "perdida_pp",
                    "cobertura_restante_pct", "intersecciones_perdidas",
                    "ya_cubierto_por_otras_pct", "google_maps"])
        for q in quitadas:
            e = q.estacion
            w.writerow([q.orden, e.id, e.nombre, e.lat, e.lon, f"{q.perdida:.1f}",
                        f"{q.perdida_pct:.4f}", f"{q.cobertura_restante_pct:.2f}",
                        q.nodos_perdidos, f"{q.compartida_pct:.1f}",
                        f"https://www.google.com/maps?q={e.lat:.6f},{e.lon:.6f}"])
    with open(args.salida / "candidatas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orden", "lat", "lon", "ganancia_pp", "cobertura_acumulada_pct",
                    "intersecciones_nuevas", "google_maps"])
        for c in candidatas:
            w.writerow([c.orden, f"{c.lat:.6f}", f"{c.lon:.6f}", f"{c.ganancia_pct:.3f}",
                        f"{c.cobertura_acumulada_pct:.2f}", c.nodos_nuevos,
                        f"https://www.google.com/maps?q={c.lat:.6f},{c.lon:.6f}"])
    with open(args.salida / "estaciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "nombre", "lat", "lon", "vecina_mas_cercana_m"])
        for ests in por_nodo.values():
            for e in ests:
                v = esp[e.id]
                w.writerow([e.id, e.nombre, e.lat, e.lon, f"{v:.0f}" if math.isfinite(v) else ""])
    grafico.dibujar_cobertura(
        grafo, dist, pesos, set(por_nodo), candidatas, args.radio, args.salida / "cobertura.png",
        f"Cobertura por calle a {args.radio:.0f} m · {len(por_nodo)} estaciones"
        + (f" + {len(candidatas)} propuestas" if candidatas else ""),
        quitadas, inv,
    )
    print(f"\nArchivos en {args.salida}/: cobertura.png, candidatas.csv, estaciones.csv, redundantes.csv")


if __name__ == "__main__":
    main()