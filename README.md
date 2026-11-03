# TP Algoritmos 2: Dijkstra sobre la red vial de CABA + estaciones CityBikes

Prototipo. Grafo dirigido de calles (OpenStreetMap), estaciones de Ecobici (API de CityBikes)
asociadas a su nodo más cercano, y Dijkstra propio con `heapq`.

## Archivos
- `grafo.py`: `Grafo` (lista de adyacencia dirigida y pesada), `Estacion`, `haversine`.
- `dijkstra.py`: `dijkstra` (con corte en el primer objetivo), `reconstruir_camino`, `camino_minimo`.
- `datos.py`: descarga/caché de estaciones y red vial, `ubicar_estaciones`, `ciudad_sintetica`.
- `main.py`: ruta por calle y estación más cercana (por calle vs. en línea recta).
- `tests/test_dijkstra.py`: validación contra `networkx`.

## Uso
```
pip install -r requirements.txt
python -m unittest discover -s tests -v     # validación (sin conexión)
python main.py --demo                       # ciudad sintética (sin conexión)
python main.py --descargar                  # baja datos reales y los guarda en datos/
python main.py                              # usa los datos guardados
python main.py --origen=-34.6037,-58.3816 --destino=-34.5810,-58.4208
```

## Notas
- Solo se usa la **ubicación** de las estaciones, no la disponibilidad en vivo.
- Se guarda un snapshot (`datos/estaciones.json`, `datos/red_vial.json.gz`) para que los resultados sean reproducibles.
- La descarga con datos reales no pudo probarse en el entorno donde se escribió el código
  (sin acceso a CityBikes/OSM); sí se probó con mocks y con la ciudad sintética.
