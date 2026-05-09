# feature: persistence-aware gating calibration

## 0) Qué significa toda esta propuesta (visión completa)

La idea general es hacer que el sistema decida mejor **cuándo confiar en el modelo tabular** y **cuándo no**.

Ahora mismo, por lo que describes, tenéis una chain que produce probabilidades para los regímenes de volatilidad. A partir de esas probabilidades decidís una clase final. El problema es que en series temporales de regímenes, sobre todo en volatilidad, hay una verdad incómoda:

- muchas veces el régimen no cambia
- y una regla muy simple como **persistence** ("mañana será igual que hoy") ya funciona muy bien

Entonces, un modelo complejo puede fallar no porque sea malo en general, sino porque sobre-reacciona y detecta cambios de régimen donde realmente no los hay.

La propuesta intenta corregir eso con tres ideas:

- calibrar mejor las probabilidades
- usar esas probabilidades calibradas para decidir si confiar en el modelo
- si no hay suficiente confianza, caer a persistence

## 1) Qué es calibrar probabilidades

Cuando un clasificador multiclase te da algo como:

- baja vol: 0.10
- media vol: 0.75
- alta vol: 0.15

uno tiende a leer eso como:

"el modelo cree con un 75% de probabilidad que la clase correcta es media vol"

Pero muchas veces eso no es verdad en sentido estadístico.

Puede ocurrir que cada vez que el modelo dice "0.75", en realidad solo acierte un 55% de las veces. O que cuando dice "0.55", en realidad acierte un 70%.

Eso significa que las probabilidades están mal calibradas.

La calibración busca aprender una transformación del estilo:

`p_raw -> p_calibrated`

para que los números finales se parezcan más a probabilidades reales.

### Ejemplo intuitivo

Imagina que el modelo predice "alta volatilidad" con estas confianzas máximas:

- predicciones con 0.90 de confianza -> solo aciertan 65%
- predicciones con 0.70 -> aciertan 60%
- predicciones con 0.55 -> aciertan 52%

Ahí el modelo está sobreconfiado.

Después de calibrar, quizás ese 0.90 pase a 0.67, el 0.70 a 0.61, etc. No necesariamente cambia qué clase va primera, pero sí cambia cuánto te puedes fiar de esa probabilidad.

Y en vuestro caso eso es fundamental, porque vais a hacer decisiones tipo:

- "si la confianza es baja, uso persistence"
- "si la confianza es alta, uso chain"
- "si está en zona intermedia, hago blend"

Si la confianza está mal medida, esas decisiones salen mal.

## 2) Qué son Platt e Isotonic

Son dos formas de hacer calibración.

### Platt scaling

Platt es un método simple y paramétrico. Aprende una transformación con forma de sigmoide:

`p = 1 / (1 + exp(-(A*x + B)))`

donde `x` es el score original del modelo y `A, B` se aprenden en validación.

Intuición:

"Voy a corregir las probabilidades con una curva suave en forma de S".

Ventajas:

- estable
- suele ir bien con pocos datos
- menos riesgo de sobreajuste

Inconveniente:

- si la descalibración real es compleja, puede quedarse corto

### Isotonic regression

Isotonic es un método más flexible y no paramétrico.

No impone una sigmoide. Solo exige que la función calibradora sea monótona creciente:

- si el score original sube, la probabilidad calibrada no puede bajar

Pero fuera de eso, la forma puede adaptarse mucho a los datos.

Intuición:

"No voy a forzar una forma concreta; dejo que los datos me digan cómo corregir las probabilidades, siempre respetando el orden".

Ventajas:

- más flexible
- puede calibrar mejor cuando hay suficientes datos

Inconvenientes:

- con pocos datos puede sobreajustar
- fold a fold puede ser menos estable que Platt

### Regla práctica entre ambos

- **Platt**: mejor cuando tienes pocos datos por fold o quieres algo robusto
- **Isotonic**: mejor cuando tienes suficientes observaciones y crees que la descalibración no es simplemente sigmoidal

En validación walk-forward, muchas veces Platt aguanta mejor cuando cada fold no es enorme. Isotonic puede funcionar mejor si tenéis bastante tamaño de validación y queréis máxima flexibilidad.

## 3) Qué significa "aplicar calibración antes de decidir clase, blend o gating"

Significa que el orden correcto del pipeline será:

1. el modelo produce scores/probabilidades brutas
2. calibráis esas probabilidades
3. con las probabilidades ya calibradas decidís:
   - la clase final
   - si se mezcla con otra señal
   - si se usa persistence o chain

Esto es importante porque si haces gating con probabilidades mal calibradas, el sistema se vuelve incoherente.

Ejemplo:

- score bruto: 0.88
- calibrado: 0.62

Sin calibrar pensarías: "muy seguro, dejo al modelo decidir".
Con calibración dirías: "no está tan seguro; quizá mejor persistence".

## 4) Qué es persistence en este contexto

Persistence es el baseline más simple posible:

"predigo que el régimen en `t` será el mismo que en `t-1`"

Por ejemplo:

- si ayer estabas en baja vol -> hoy predigo baja vol
- si ayer estabas en alta vol -> hoy predigo alta vol

En regímenes de volatilidad esto suele ser fortísimo porque los estados son persistentes. Muchas observaciones son **non-transition rows**, es decir, filas donde el régimen no cambia.

Por eso un modelo complejo puede parecer bueno, pero si no supera claramente a persistence, en realidad no está aportando demasiado.

## 5) Qué es el gating contra persistence

El gating es una regla de decisión superior que elige entre dos fuentes:

- la chain (vuestro sistema complejo)
- persistence (el baseline simple)

La lógica sería:

```python
if confidence < threshold:
    pred = persistence
else:
    pred = chain
```

Traducido:

- si el modelo no está suficientemente seguro, no se le hace caso
- si el modelo sí está suficientemente seguro, se usa su predicción

Intuición real:

"No quiero que el modelo me invente cambios de régimen cuando no tiene evidencia clara".

Eso ataca un fallo muy común en clasificación temporal: la sobre-reacción.

Un clasificador potente puede detectar transiciones reales, sí, pero también puede tener tendencia a "ver" demasiadas. Persistence, en cambio, suele ser aburrido pero muy robusto en tramos estables.

El gating intenta quedarse con lo mejor de ambos:

- estabilidad de persistence cuando no hay señal clara
- capacidad del modelo para detectar transiciones cuando sí la hay

## 6) Qué es el blend y por qué importa la calibración

Además del gating duro, puede haber un blend entre señales.

Eso significa mezclar, por ejemplo:

- la distribución de probabilidad de la chain
- la distribución implícita o preferencia por persistence
- con algún peso `alpha`

Conceptualmente:

`p_final = alpha * p_chain + (1 - alpha) * p_persistence`

Si además el blend depende de la confianza, entonces entra también algo tipo `blend_conf_beta`, que modula cuánto cambia la mezcla cuando el modelo parece muy seguro o poco seguro.

Pero aquí vuelve el mismo problema: si las probabilidades no están calibradas, mezclar según confianza tiene poco sentido.

## 7) Qué se va a tunear exactamente

En validación walk-forward vais a buscar los mejores valores para varios hiperparámetros.

### threshold

Es el umbral de confianza para el gating.

Ejemplo:

- si `threshold = 0.65`
- y la confianza calibrada máxima es `0.58`
- entonces usas persistence

Si la confianza es `0.72`, usas chain.

### blend_alpha

Controla cuánto peso se da a cada señal en la mezcla.

- alpha alto -> más peso al modelo
- alpha bajo -> más peso a persistence

### blend_conf_beta

Controla cuánto influye la confianza en esa mezcla.

Intuitivamente:

- con beta alto, pequeñas diferencias de confianza cambian mucho el blend
- con beta bajo, el blend cambia más suavemente

### thresholds por clase

Esto es muy útil en problemas desbalanceados.

No todas las clases son igual de frecuentes ni igual de importantes.

Ejemplo típico:

- baja volatilidad: frecuente
- media volatilidad: frecuente
- alta volatilidad: menos frecuente pero muy importante

Entonces quizá no quieras la misma exigencia de confianza para todas.

Por ejemplo:

- para predecir alta vol puedes aceptar una confianza más baja si tu objetivo es subir recall
- o exigir una confianza más alta si quieres evitar falsas alarmas

Eso depende de vuestra métrica objetivo.

## 8) Por qué thresholds por clase pueden ayudar mucho

En clasificación de regímenes, la clase rara suele ser la que más interés económico tiene.

Por ejemplo, la alta volatilidad:

- puede estar poco representada
- pero es la que más importa para gestión de riesgo, sizing, drawdowns, etc.

Entonces optimizar solo una regla global puede perjudicar justo la clase importante.

Los thresholds por clase permiten decir algo así:

- "Para baja vol soy conservador"
- "Para alta vol prefiero mayor sensibilidad"

Eso puede mejorar mucho métricas como:

- macro F1
- recall de alta vol
- balanced accuracy

## 9) Qué significa reportar métricas en all, transition rows y non-transition rows

Esto es probablemente de las partes más sensatas de toda la propuesta.

### all

Métricas sobre todas las filas.

Te da la foto global, pero puede ocultar problemas importantes.

### non-transition rows

Filas donde el régimen no cambia.

Aquí persistence suele ser muy fuerte.

Si tu sistema mejora poco o empeora aquí, eso es una señal de que quizá el modelo está metiendo ruido innecesario.

### transition rows

Filas donde sí hay cambio de régimen.

Aquí es donde los modelos complejos deberían aportar valor. Si no mejoran frente a persistence aquí, entonces casi no merece la pena su complejidad.

### delta vs persistence

Comparar explícitamente contra persistence.

Esto es clave porque en este tipo de problemas puedes obtener métricas "decentes" y, aun así, estar por debajo de una regla trivial.

El delta vs persistence te dice:

"¿Estamos mejorando de verdad respecto al baseline fuerte, o solo nos estamos engañando con un modelo más sofisticado?"

## 10) Qué quiere decir "mantener compatibilidad total detrás de flags"

Significa que no vais a romper el pipeline actual.

Todo lo nuevo se activará opcionalmente con flags:

- calibración activada o no
- tipo de calibrador
- gating activado o no
- blend activado o no
- thresholds por clase activados o no

Y con defaults conservadores.

Eso permite:

- hacer A/B tests limpios
- comparar contra baseline
- evitar que el sistema cambie de comportamiento por accidente
- desplegar gradualmente

Es justo lo correcto para trabajo experimental serio.

## 11) Por qué esta estrategia puede mejorar más que "meter más modelo"

La intuición de fondo es muy potente:

en out-of-sample muchas veces gana más una mejor lógica de decisión que un modelo más complejo.

Razón 1: persistence ya es fortísimo.

Cuando el mercado está en un régimen estable, persistence es difícil de batir. Si fuerzas al chain a decidir siempre, puede introducir falsos cambios.

El gating reduce eso.

Razón 2: el problema muchas veces no es el ranking, sino la confianza.

A veces el modelo ordena razonablemente bien las clases, pero las probabilidades están mal.

Entonces:

- con probabilidades crudas tomas malas decisiones de umbral
- con probabilidades calibradas el mismo modelo se vuelve más útil

Razón 3: más complejidad no siempre da más robustez.

Puedes complicar el modelo base y ganar algo in-sample o en un fold concreto, pero perder estabilidad out-of-sample.

En cambio:

- calibración
- fallback a persistence
- tuning por fold
- reporting por transitions

suele mejorar robustez, que para series temporales vale oro.

## 12) Cómo se vería fila a fila en inferencia

Imagina una fila donde el régimen previo es media vol.

La chain produce:

- baja: 0.18
- media: 0.47
- alta: 0.35

Sin calibración y sin gating:

- escoges media, porque 0.47 es la mayor

Con calibración:

Tras calibrar puede quedar:

- baja: 0.16
- media: 0.41
- alta: 0.43

Ahora ya no está tan claro. Quizá pasa a ganar alta, o al menos la confianza máxima sigue siendo baja.

Con gating:

- si el threshold es 0.60 y la confianza máxima calibrada es 0.43
- el sistema decide que no hay suficiente seguridad
- usa persistence
- como el régimen previo era media vol, predice media vol

Eso puede evitar una falsa transición a alta vol.

Ahora otro caso:

- régimen previo: media vol
- la chain produce, tras calibración:
  - baja: 0.05
  - media: 0.22
  - alta: 0.73

Como 0.73 supera el umbral:

- el sistema sí confía en la chain
- predice alta vol

Aquí permites detectar una transición real.

## 13) Cómo encajan Platt e Isotonic exactamente en vuestro flujo

La secuencia correcta sería:

1. entrenas el modelo base en el fold train
2. ese modelo produce probabilidades en validación
3. con esas predicciones de validación ajustas el calibrador (Platt o Isotonic), siempre sin mirar test futuro
4. aplicas el calibrador a las probabilidades del fold correspondiente
5. con esas probabilidades calibradas:
   - decides clase
   - haces blend
   - aplicas gating
   - comparas con persistence
6. eliges hiperparámetros por fold (thresholds, alpha, beta, etc.)

Esto evita leakage y respeta la lógica walk-forward.

## 14) Qué peligro evita la calibración en vuestro caso

Imagina que el modelo está estructuralmente sobreconfiado:

- muchas predicciones salen con max prob 0.80-0.95
- pero la tasa real de acierto en esa banda es mucho menor

Entonces un gating basado en "si > 0.70 confío" será un desastre:

- casi siempre confiarás en la chain
- apenas usarás persistence
- perderás justo el beneficio que querías

La calibración corrige eso y hace que el threshold tenga significado.

## 15) Qué os están proponiendo realmente a nivel de filosofía

No os están diciendo:

"vamos a entrenar un modelo radicalmente mejor"

Os están diciendo algo más interesante:

"vamos a construir una capa de decisión más inteligente encima del modelo actual"

Eso es muy distinto.

En vez de gastar toda la mejora esperada en arquitectura, la gastáis en:

- fiabilidad de probabilidades
- decisión condicional
- fallback robusto
- tuning según el objetivo real
- evaluación separada por tipo de caso

Eso suele ser muy buena ingeniería de ML temporal.

## 16) Resumen ejecutivo en una frase por componente

### Calibración

Corrige las probabilidades para que el nivel de confianza del modelo sea creíble.

### Platt

Calibración suave con sigmoide; más estable, menos flexible.

### Isotonic

Calibración monótona flexible; más potente si hay suficientes datos.

### Gating

Si el modelo no está seguro, no se le hace caso y se usa persistence.

### Persistence

Baseline que predice que el régimen sigue igual que el anterior; muy fuerte en tramos estables.

### Threshold tuning

Busca cuánta confianza mínima hay que exigir para dejar hablar al modelo.

### Thresholds por clase

Permiten tratar distinto clases desbalanceadas o especialmente importantes, como alta vol.

### Reporting por transitions

Sirve para ver si mejoráis donde importa: cambios reales de régimen frente a estabilidad.

## 17) Resumen final muy en cristiano

Lo que vais a implementar es un sistema que funcione así:

1. el modelo predice probabilidades de régimen
2. esas probabilidades se corrigen para que no "mientan" sobre la confianza
3. si el modelo está realmente seguro, se usa su predicción
4. si no lo está, se usa la regla simple de que el régimen continúa
5. todo eso se ajusta fold a fold y se evalúa separando los casos fáciles de los difíciles

La apuesta de fondo es:

en este problema, mejorar la toma de decisiones sobre la salida del modelo puede dar más robustez real que complicar más el modelo base.

Y, sinceramente, para predicción de regímenes de volatilidad, esa es una apuesta bastante sensata.

## 18) Implementación realizada en código (estado actual de la rama)

Esta sección documenta lo que ya está implementado de forma efectiva en `feature/persistence-aware-gating-calibration`.

### 18.1 Cambios core en `scripts/walk_forward_chain_tab.py`

Se han añadido y conectado de extremo a extremo los siguientes bloques:

- **Calibración de probabilidades**:
  - `_fit_probability_calibrator(...)` con soporte `none|platt|isotonic`.
  - `_apply_probability_calibrator(...)` para aplicar calibración en valid/test.
  - Normalización robusta de filas de probabilidad con `_normalize_proba_rows(...)`.

- **Predicción con thresholds por clase**:
  - `_predict_with_class_thresholds(...)`.
  - Parsing/formateo:
    - `_parse_class_threshold_grid(...)`
    - `_parse_class_thresholds(...)`
    - `_format_class_thresholds(...)`

- **Gating persistence-aware**:
  - `_blend_strategy_predict(...)`:
    - mezcla `alpha/beta` entre chain y persistence en espacio de probabilidad
    - confianza de mezcla
    - fallback a persistence cuando `confidence < gate_threshold`

- **Selección conjunta de estrategia**:
  - `_pick_best_blend_strategy(...)` optimiza de forma conjunta:
    - `alpha`
    - `beta`
    - `class_thresholds`
    - `gate_threshold`

- **Métricas extendidas**:
  - además de Accuracy/F1/Recall, se guardan:
    - `transition_macro_f1` y delta vs persistence
    - `non_transition_macro_f1` y delta vs persistence
    - `gating_rate`, `mean_confidence`

### 18.2 Integración en el loop walk-forward

Se ha integrado la lógica anterior en los puntos clave:

- `_fit_eval_tabular(...)`:
  - calibra probas (train->fit calibrador)
  - evalúa valid con probas calibradas
  - aplica tuning conjunto de blend/gating/thresholds

- selección OOF en `main()`:
  - se usa `_pick_best_blend_strategy(...)` sobre el OOF consolidado
  - se recalculan métricas fold con la estrategia OOF seleccionada
  - se añade `transition_bonus_lambda` al `robust_score`

- `_official_test_compare(...)`:
  - ajusta calibrador en valid del split oficial
  - aplica calibración en test
  - aplica estrategia seleccionada (`alpha/beta/class_thresholds/gate_threshold`)
  - reporta también métricas de transición/no transición

### 18.3 Nuevos flags CLI añadidos

En `scripts/walk_forward_chain_tab.py` se añadieron:

- `--calibration_method {none,platt,isotonic}`
- `--gate_thresholds "v1,v2,..."`
- `--class_threshold_grid "a,b,c;d,e,f;..."`
- `--transition_bonus_lambda float`

Se mantienen intactos los flags existentes de blend (`--use_blend`, `--blend_alphas`, `--blend_conf_betas`) para compatibilidad.

### 18.4 Salida de reporting ampliada

Ahora en `fold_metrics.csv`, `summary.csv` y `final_vs_persistence.csv` aparecen campos adicionales:

- configuración:
  - `selected_class_thresholds`
  - `selected_gate_threshold`
  - `selected_calibration_method`
- comportamiento:
  - `mean_confidence_*`
  - `gating_rate_*`
- transición:
  - `*_transition_macro_f1*`
  - `*_non_transition_macro_f1*`
  - deltas correspondientes vs persistence

### 18.5 Tests añadidos y validaciones ejecutadas

Se añadió el smoke test:

- `tests/test_chain_gating_calibration_smoke.py`
  - valida routing de gating a persistence en baja confianza
  - valida que thresholds por clase cambian decisión
  - valida calibración Platt y normalización final de probas

Ejecuciones realizadas:

- `poetry run pytest -q tests/test_chain_gating_calibration_smoke.py`
- `poetry run pytest -q tests/test_chain_gating_calibration_smoke.py tests/test_har_rv_smoke.py tests/test_skewt_smoke.py`
- smoke CLI con blend+gating+calibración:
  - `runs/_tmp_gating_smoke/...`
- smoke CLI sin blend (compatibilidad):
  - `runs/_tmp_gating_smoke_noblend/...`

Resultado de validación: sin errores funcionales en el flujo completo.
