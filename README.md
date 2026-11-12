# TP Algoritmos 2: Dijkstra sobre la red vial de CABA + estaciones CityBikes

Grafo dirigido de calles (OpenStreetMap), estaciones de bicicletas públicas (API de CityBikes)
asociadas a su nodo más cercano, y Dijkstra propio con `heapq`. Tres usos:

1. **Rutas** (`main.py`): camino más corto por calle entre dos puntos y estación más cercana.
2. **Cobertura y nuevas estaciones** (`analizar.py`): qué tan bien cubre la red actual cada zona
   de la ciudad, dónde conviene poner estaciones nuevas y cuáles sobran.
3. **Comparación de criterios** (`comparar.py`): ubica el mismo número de estaciones nuevas con tres
   criterios distintos (máxima cobertura, p-mediana, p-centro) y los evalúa con las mismas métricas.

## Archivos
- `grafo.py`: `Grafo` (lista de adyacencia dirigida y pesada), `Estacion`, `haversine`,
  `invertido()`, `subgrafo()`, `componente_fuerte_mayor()` (Kosaraju iterativo).
- `dijkstra.py`: `dijkstra` (corte en el primer objetivo y/o radio máximo), `dijkstra_multifuente`,
  `reconstruir_camino`, `camino_minimo`.
- `datos.py`: descarga/caché de estaciones y red vial (con reintentos), `ubicar_estaciones`, `ciudad_sintetica`.
- `analisis.py`: cobertura, espaciado entre estaciones, selección de ubicaciones nuevas.
- `comparacion.py`: `p_mediana`, `p_centro`, `evaluar`, `comparar_objetivos`, `coincidencias`.
- `preparacion.py`: carga y preparación de datos compartida por `analizar.py` y `comparar.py`.
- `grafico.py`: mapas e histograma. `analizar.py`: cobertura. `comparar.py`: comparación. `main.py`: rutas.
- `tests/`: validación contra `networkx` y fuerza bruta.

## Uso
```
pip install -r requirements.txt
python -m unittest discover -s tests -v     # validación (sin conexión)
python main.py --descargar                  # baja datos reales y los guarda en datos/
python main.py --demo                       # ruta de ejemplo; genera resultados/ruta.png
python analizar.py                          # análisis con los datos guardados
python analizar.py --radio 400 --k 15       # radio de cobertura (m) y estaciones nuevas
python analizar.py --quitar 20              # cuántas estaciones redundantes listar (por defecto 10)
python analizar.py --demanda puntos.csv     # ponderar por demanda (CSV con lat,lon[,peso])
python analizar.py --demo                   # todo con una ciudad sintética, sin conexión
```
Salida de `analizar.py` en `resultados/`: `cobertura.png`, `candidatas.csv` y `redundantes.csv`
(ambos con link a Google Maps por ubicación) y `estaciones.csv`.

## Gráficos
- `ruta.png` (`main.py`): la ruta por calle, la línea recta, las estaciones más cercanas (por calle y en
  recta) y la "ola" de Dijkstra: los nodos que exploró hasta fijar el destino, coloreados por distancia.
- `cobertura.png` (`analizar.py`): mapa de distancia a la estación más cercana (la línea en la barra de color
  marca el radio), curva de cobertura acumulada **hoy vs. con las propuestas**, y barras con lo que suma cada
  ubicación propuesta y lo que se pierde al sacar cada estación redundante.
- `comparacion.png` (`comparar.py`): un mapa por criterio con la distancia que resulta de aplicarlo (misma
  escala en los tres) y una columna por métrica; ✓ marca el mejor criterio en cada una.

Comparación de criterios:
```
python comparar.py --k 20 --radio 400       # genera comparacion.png y dos CSV en resultados/
python comparar.py --demo --k 8 --radio 250 # con la ciudad sintética, sin conexión
```

## Cómo funciona el análisis
- **Distancia que importa:** metros *por calle* desde un punto hasta la estación más cercana, respetando
  los sentidos de las calles. Un solo Dijkstra multi-fuente (todas las estaciones con distancia 0) sobre el
  grafo **invertido** la calcula para todos los nodos: mismo costo que un Dijkstra común, en vez de uno por estación.
- **Cubierto** = a `--radio` metros o menos (400 por defecto). Se informa para 300/400/500/600 m.
- **Nuevas estaciones:** greedy de máxima cobertura con evaluación perezosa. Cada paso elige la intersección
  que cubre más peso hoy descubierto. Es aproximado, con garantía de al menos ~63 % del óptimo
  (la cobertura es submodular).
- **Estaciones redundantes:** es el problema espejo, un greedy inverso. Cada paso saca la estación que
  hace perder menos cobertura *en exclusiva*: una intersección cubierta por 2 o más estaciones no se
  pierde al sacar una. Como al ir sacando la pérdida de las demás solo puede crecer, alcanza con
  evaluación perezosa. Se informa la pérdida, la cobertura restante y qué % de lo que cubre cada
  estación ya lo cubren otras.
- **Tres criterios de ubicación** (`comparar.py`), todos partiendo de las estaciones actuales:
  - *Máxima cobertura*: el greedy de arriba, para un radio fijo.
  - *p-mediana*: minimiza la distancia **media** ponderada. Greedy perezoso: la ganancia de un candidato
    se calcula con una búsqueda podada (`dijkstra_podado`) que solo explora donde ese candidato mejoraría
    la distancia actual; la poda es exacta por la desigualdad triangular.
  - *p-centro*: minimiza la distancia **máxima**, con la heurística del punto más lejano.

  Los tres se evalúan igual, recalculando desde cero con el Dijkstra multi-fuente, y se informa qué
  tanto coinciden en los lugares elegidos (dos ubicaciones a menos de 2 × radio solapan su zona).
- **Peso de cada punto:** metros de calle que salen de la intersección, o la demanda real si se pasa `--demanda`.
- Solo se analiza la **mayor componente fuertemente conexa** de la red: los tramos aislados o sin salida
  se descartan, porque ahí "distancia a una estación" no tiene sentido.

## Limitaciones
- Sin `--demanda`, el criterio es solo de **oferta** (zonas lejos de estaciones), no de demanda: puede proponer
  zonas poco pobladas. Cruzarlo con población, comercios o transporte mejora mucho el resultado.
- **p-centro es muy sensible a los extremos:** mide la ciudad por su peor rincón, así que tiende a elegir
  bordes y esquinas (parques, predios, zonas casi sin gente). Con datos de demanda esto se corrige, porque
  los nodos sin peso no cuentan. La heurística tampoco es óptima: en pruebas puede quedar apenas por
  detrás de p-mediana incluso en su propia métrica.
- Una estación "redundante" lo es **para la cobertura**, no necesariamente para el servicio: la cobertura
  no mide capacidad, cantidad de bicis ni uso. Una estación solapada puede ser la que absorbe la demanda.
- Las ubicaciones son **intersecciones**: hay que verificar en el lugar si hay espacio (vereda, permiso).
- Solo se usa la **ubicación** de las estaciones, no la disponibilidad en vivo ni los viajes reales.
- Se guarda un snapshot (`datos/estaciones.json`, `datos/red_vial.json.gz`) para que los resultados sean reproducibles.
