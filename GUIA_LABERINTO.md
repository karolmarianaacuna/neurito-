# Neurito 3.0: aprende a encontrar comida en un laberinto

Neurito debe elegir entre cuatro movimientos: arriba, abajo, izquierda y derecha. El programa combina cuatro piezas:

- **El mundo** guarda los mapas, la posici?n de Neurito, la comida y las paredes (`laberinto_mundo.py`).
- **BFS, el experto**, calcula qu? movimiento acerca a Neurito a la comida y genera las respuestas correctas para los ejemplos.
- **La red neuronal** aprende a imitar esas respuestas (`laberinto_red.py`).
- **La interfaz** muestra el laberinto y permite entrenar y jugar (`laberinto.py`).

La red no descubre las rutas por s? sola. Aprende de ejemplos creados por BFS. Esto se llama **aprendizaje supervisado**: damos entradas y respuestas esperadas para que el modelo aprenda a relacionarlas. No es aprendizaje por refuerzo, porque el programa no aprende a partir de premios o castigos recibidos durante el juego.

## C?mo representa la situaci?n

La red recibe 165 n?meros: un identificador del mapa, la posici?n de Neurito y la posici?n de la comida. Las posiciones se codifican con vectores *one-hot*: casi todos sus valores son cero y el valor correspondiente a la casilla ocupada es uno.

En `laberinto_red.py`:

```python
LADO = 9
ENTRADAS = len(MAPAS) + 2 * LADO * LADO  # 3 + 81 + 81 = 165
```

La funci?n `codificar()` prepara esos n?meros:

```python
def codificar(indice, posicion, meta):
    x = np.zeros(ENTRADAS, dtype=np.float64)
    x[indice] = 1
    n = len(MAPAS)
    x[n + posicion[0] * LADO + posicion[1]] = 1
    x[n + LADO * LADO + meta[0] * LADO + meta[1]] = 1
    return x
```

`indice` identifica cu?l mapa se usa; `posicion` es la casilla de Neurito; `meta` es la casilla de la comida. La red no recibe una imagen ni una descripci?n directa de todas las paredes: conoce el identificador de uno de los tres mapas usados para crear los datos.

La arquitectura tiene 165 entradas, 96 neuronas ocultas y 4 salidas. Las salidas representan las cuatro direcciones. La capa oculta usa ReLU y las salidas usan Softmax, que convierte cuatro n?meros en probabilidades que suman 1 (o 100 %).

## 1. ?Por qu? se atasca Neurito sin entrenar?

Al crear la red, sus pesos empiezan con valores aleatorios. Los pesos son n?meros que determinan c?mo la informaci?n de entrada influye en las decisiones. Antes de aprender, las probabilidades no tienen por qu? favorecer el movimiento que lleva a la comida.

En `laberinto_red.py`, el constructor prepara el tama?o de la red y llama a `reiniciar()`:

```python
def __init__(self, semilla=42, ocultas=96, tasa=0.45):
    self.rng = np.random.default_rng(semilla)
    self.ocultas = ocultas
    self.tasa = tasa
    self.reiniciar()
```

`reiniciar()` crea pesos iniciales para las conexiones:

```python
def reiniciar(self):
    self.W1 = self.rng.normal(0, np.sqrt(2 / ENTRADAS), (ENTRADAS, self.ocultas))
    self.b1 = np.zeros((1, self.ocultas))
    self.W2 = self.rng.normal(0, np.sqrt(2 / self.ocultas), (self.ocultas, 4))
    self.b2 = np.zeros((1, 4))
    self.epocas = 0
    self.historial = []
```

`W1` y `W2` son matrices de pesos; `b1` y `b2` son sesgos que tambi?n ajustan los c?lculos. No hace falta memorizar las matrices: lo importante es que sus valores se modifican durante el entrenamiento.

Al jugar, la red calcula las probabilidades y se escoge la mayor. Antes del entrenamiento esa elecci?n puede ser mala. Adem?s, jugar no entrena la red autom?ticamente. Las teclas `1` y `2` inician el entrenamiento; `3` lo activa de forma continua.

**En resumen:** Neurito se atasca porque sus pesos a?n no reflejan lo que ense?an los ejemplos y, en red pura, puede elegir una direcci?n que choca con una pared o repetir un movimiento.

## 2. ?Qu? ocurre al entrenar 100, 200 y 400 ?pocas?

Una **?poca** es una pasada de aprendizaje sobre todos los ejemplos de entrenamiento. En cada lote, la red calcula sus predicciones, las compara con las respuestas de BFS y ajusta sus pesos para reducir los errores.

La tecla `2` ejecuta cien ?pocas. En `laberinto.py`:

```python
elif tecla == pygame.K_2:
    for _ in range(100):
        red.entrenar_epoca(X, y)
```

Pulsa `2` una vez para 100 ?pocas, dos veces para 200 y cuatro veces para 400. En la pantalla puedes observar la p?rdida y la precisi?n de prueba; luego observa c?mo juega Neurito.

### Qu? significa cada medida

La **p?rdida** resume qu? tan alejadas est?n las predicciones de las respuestas correctas durante el entrenamiento. `entrenar_epoca()` la calcula y guarda el resultado:

```python
perdida = total / len(X)
self.historial.append(float(perdida))
return perdida
```

Si la p?rdida baja con el entrenamiento, normalmente significa que la red est? ajust?ndose mejor a los ejemplos. No mide directamente si completa laberintos.

La **precisi?n de prueba** es la proporci?n de ejemplos reservados para prueba en los que la direcci?n m?s probable coincide con la respuesta esperada:

```python
def exactitud(self, X, y):
    return float(np.mean(np.argmax(self.adelante(X)[2], axis=1) == y))
```

`adelante(X)[2]` son las probabilidades de las cuatro direcciones. `argmax(...)` elige la direcci?n m?s probable. La comparaci?n `== y` cuenta los aciertos, y `mean(...)` calcula su proporci?n.

Los **movimientos** muestran el resultado completo: si llega a la comida, cu?ntos pasos tarda, si choca o si repite casillas. Una sola ruta requiere muchas decisiones correctas seguidas. Por eso la precisi?n puede ser alta y, aun as?, Neurito equivocarse en un paso importante y no llegar.

Anota las tres medidas en cada punto:

| Entrenamiento | P?rdida | Precisi?n de prueba | Resultado del juego |
|---|---|---|---|
| 100 ?pocas |  |  |  |
| 200 ?pocas |  |  |  |
| 400 ?pocas |  |  |  |

No inventes de antemano un resultado: ejec?talo y completa la tabla. Es com?n que la p?rdida baje y la precisi?n suba, pero no est? garantizado que cada tanda mejore todos los recorridos.

## 3. ?Qu? diferencia hay entre red pura y navegaci?n asistida?

En **red pura**, la red calcula cuatro probabilidades y se elige la m?s alta. En **modo asistido**, se usan esas mismas probabilidades, pero unas reglas adicionales descartan movimientos hacia paredes y penalizan casillas ya visitadas.

En `laberinto_mundo.py`, la decisi?n se implementa as?:

```python
def elegir_accion(red, mundo, visitas, asistido):
    probs = red.probabilidades(mundo.indice, mundo.posicion, mundo.meta)

    if not asistido:
        return int(probs.argmax()), probs

    opciones = [
        (float(probs[a]) - 0.35 * visitas.get(vecino(mundo.posicion, a), 0), -a, a)
        for a in range(4) if valido(mundo.mapa, vecino(mundo.posicion, a))
    ]
    return max(opciones)[2], probs
```

Paso a paso:

1. `probs = ...` pide a la red sus cuatro probabilidades.
2. Si `asistido` es falso, `argmax()` devuelve el ?ndice de la probabilidad mayor. Eso es red pura.
3. Si est? asistido, `valido(...)` deja fuera las direcciones que llevan a una pared.
4. `visitas.get(..., 0)` cuenta cu?ntas veces se visit? la casilla vecina; el programa le resta `0.35` por cada visita.
5. `max(opciones)` selecciona la opci?n con mejor puntuaci?n ajustada.

La tecla `A` cambia entre ambos modos. La red aporta las probabilidades; el filtro de paredes y la penalizaci?n de visitas son reglas programadas, no aprendizaje neuronal. La asistencia puede ayudar, pero no garantiza encontrar la comida: la penalizaci?n local no calcula una ruta completa.

## 4. ?Qu? pasa al cambiar las 96 neuronas ocultas por 16, 32 o 128?

El tama?o de la capa oculta indica cu?ntas unidades intermedias usa la red para relacionar la entrada con las cuatro salidas. El valor actual se define en el constructor de `RedLaberinto` en `laberinto_red.py`:

```python
class RedLaberinto:
    def __init__(self, semilla=42, ocultas=96, tasa=0.45):
```

La variable `ocultas` tambi?n define el tama?o de las matrices de pesos. Al cambiarla, la red se crea con otra cantidad de neuronas.

- Con **16**, tiene menos capacidad para representar patrones. Podr?a aprender peor, aunque depende de los datos y el entrenamiento.
- Con **32**, dispone de m?s capacidad que con 16.
- Con **128**, tiene m?s unidades que la configuraci?n actual. Eso no asegura mejor precisi?n ni mejores recorridos.

Para hacer la comparaci?n, crea la red con `ocultas=16`, luego `32`, `96` y `128`. Mant?n iguales los datos, la tasa y las ?pocas. Anota p?rdida, precisi?n y recorridos. Si cambias varias cosas a la vez, no sabr?s cu?l caus? la diferencia.

## 5. ?Qu? hace la tasa de aprendizaje 0.45?

La tasa de aprendizaje controla el tama?o de los cambios a los pesos. El valor predeterminado aparece en el constructor:

```python
def __init__(self, semilla=42, ocultas=96, tasa=0.45):
    self.tasa = tasa
```

Dentro de `entrenar_epoca()`, se multiplica por los cambios calculados para cada matriz:

```python
self.W1 -= self.tasa * dw1
self.b1 -= self.tasa * db1
self.W2 -= self.tasa * dw2
self.b2 -= self.tasa * db2
```

`dw1`, `db1`, `dw2` y `db2` indican hacia d?nde habr?a que ajustar los pesos y sesgos para reducir el error. La tasa hace el ajuste m?s grande o m?s peque?o.

- Una tasa **muy peque?a** puede hacer que la p?rdida baje lentamente.
- Una tasa **demasiado grande** puede hacer que los ajustes se pasen; la p?rdida puede oscilar o subir.
- Una tasa **adecuada** suele producir una bajada m?s estable, aunque no garantiza que el juego resuelva todos los retos.

Prueba una tasa a la vez. Observa la p?rdida despu?s de cada tanda y compara tambi?n precisi?n y recorridos. Decimos que el aprendizaje parece converger cuando la p?rdida se estabiliza en un valor bajo; no basta con que una partida haya salido bien.

## 6. ?Qu? significa cambiar de mapa si los tres mapas forman parte del entrenamiento?

Los tres mapas est?n definidos en `MAPAS`, en `laberinto_mundo.py`. Al generar ejemplos, el programa considera posiciones y metas posibles en esos mapas. Despu?s mezcla los ejemplos y aparta un 20 % para prueba:

```python
rng.shuffle(muestras)
corte = int(len(muestras) * (1 - fraccion_prueba))
```

El 80 % se usa para entrenar y el resto para medir la precisi?n. Los pares de posici?n y meta de prueba no se repiten en los datos de entrenamiento. La tecla `N` cambia el mapa durante el juego.

Esto comprueba si la red generaliza a **nuevas combinaciones de inicio y meta en mapas conocidos**. No comprueba si reconoce un laberinto con paredes que nunca vio: el modelo recibe el identificador de uno de los tres mapas y la representaci?n actual no codifica las paredes de un mapa arbitrario.

Para probar laberintos realmente nuevos, habr?a que representar sus paredes en las entradas y separar los mapas completos: entrenar con unos mapas y evaluar con otros que la red no vio.

## 7. ?Qu? cambiar?a al sustituir BFS por DFS o A*?

BFS, DFS y A* buscan rutas, pero exploran el laberinto de formas distintas. Aqu? el buscador es el **experto** que genera las respuestas correctas para entrenar la red. Si cambias el buscador, cambias las respuestas que la red aprende a imitar.

### BFS: explorar por niveles

BFS visita primero las casillas a un paso, luego las que est?n a dos pasos, y as? sucesivamente. Cuando todos los movimientos cuestan lo mismo, encuentra una ruta con el menor n?mero de pasos. La funci?n est? en `laberinto_mundo.py`:

```python
def distancias_bfs(mapa, meta):
    dist = {meta: 0}
    cola = deque([meta])
    while cola:
        actual = cola.popleft()
        for a in range(4):
            siguiente = vecino(actual, a)
            if valido(mapa, siguiente) and siguiente not in dist:
                dist[siguiente] = dist[actual] + 1
                cola.append(siguiente)
    return dist
```

`cola` guarda las casillas que falta explorar. `dist` registra cu?ntos pasos hay desde cada casilla hasta la meta. `popleft()` saca la casilla m?s antigua de la cola, que es lo que permite explorar por niveles. `valido()` impide atravesar paredes.

Una vez calculadas las distancias, `accion_experta()` recomienda una direcci?n que reduzca la distancia:

```python
def accion_experta(mapa, origen, dist):
    if origen not in dist or dist[origen] == 0:
        return None
    opciones = [(dist.get(vecino(origen, a), 10**9), a) for a in range(4)]
    minimo, accion = min(opciones)
    assert minimo < dist[origen]
    return accion
```

`opciones` contiene, para cada direcci?n, la distancia de la casilla vecina a la meta y el n?mero de esa direcci?n. `min()` elige la distancia menor. El `assert` comprueba que el paso realmente acerque a Neurito.

### DFS: explorar en profundidad

DFS sigue una direcci?n todo lo que puede. Si llega a un callej?n sin salida, retrocede y explora otra ruta. Puede encontrar un camino, pero no garantiza que sea el m?s corto. La ruta depende del orden en que prueba arriba, abajo, izquierda y derecha.

### A*: combinar lo recorrido con lo que parece faltar

A* prioriza las rutas usando dos cantidades: el coste real desde el inicio y una estimaci?n del coste restante hasta la meta. En una cuadr?cula, la distancia Manhattan puede servir como estimaci?n:

```python
def heuristica(posicion, meta):
    return abs(posicion[0] - meta[0]) + abs(posicion[1] - meta[1])
```

Cuenta diferencias de filas y columnas, pero ignora las paredes. Con movimientos de coste uniforme y una heur?stica que no sobreestima la distancia real, A* puede encontrar una ruta ?ptima. La calidad de la heur?stica afecta cu?ntas casillas explora.

### C?mo compararlos

Cambiar BFS por DFS o A* requiere cambiar el c?digo que genera las etiquetas y volver a crear los datos y entrenar la red. Solo cambiar el nombre del algoritmo no basta. Para una comparaci?n justa, usa los mismos mapas y retos, y cambia una cosa a la vez.

Compara tres cosas:

1. **Ruta del experto:** cu?ntos pasos necesita el algoritmo para llegar a la comida.
2. **Aprendizaje:** p?rdida y precisi?n de la red entrenada con las etiquetas de ese experto.
3. **Juego:** cu?ntas veces Neurito llega, cu?ntos pasos da, y si choca o repite casillas.

Una precisi?n alta solo dice que la red imita bien las respuestas usadas como referencia. Si el experto DFS eligi? rutas largas, la red podr?a imitarlas con precisi?n y aun as? recorrer m?s pasos que una red entrenada con BFS. Por eso la precisi?n no decide por s? sola cu?l buscador produce el mejor comportamiento.

## Ejecuci?n y controles

Ejecuta el juego desde la carpeta del proyecto:

```bash
uv sync
uv run laberinto.py
```

| Tecla | Acci?n |
|---|---|
| `1` | Entrenar una ?poca |
| `2` | Entrenar cien ?pocas |
| `3` | Activar o pausar entrenamiento continuo |
| `A` | Alternar red pura y navegaci?n asistida |
| `P` | Alternar avance autom?tico y paso a paso |
| `Espacio` | Avanzar un paso en modo manual |
| `N` | Cambiar de mapa |
| `G` | Generar otra posici?n y meta |
| `C` | Reiniciar contadores |
| `R` | Reiniciar pesos y entrenamiento |
| `Esc` | Salir |

La aplicaci?n reutiliza `assets/neutral_1.png`, `assets/neutral_2.png` y `assets/neutral_3.png`. No necesita dependencias nuevas.

Las pruebas automatizadas incluidas se pueden ejecutar con:

```bash
uv run python -m unittest -v test_laberinto.py
```

## Alcance y limitaciones

El modelo aprende a elegir movimientos a partir de caracter?sticas estructuradas, no de im?genes. Los tres mapas est?n presentes al generar los ejemplos; la prueba mide generalizaci?n a pares de posici?n y meta reservados, no a mapas completamente desconocidos. La navegaci?n asistida agrega reglas simples, pero tampoco garantiza resolver todos los laberintos.
