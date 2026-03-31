# feature: chain HAR + sigma features

## 0) Objetivo de esta feature

`feature/chain-har-sigma-features` se introdujo para fortalecer el bloque econometrico del pipeline sin romper el flujo existente:

- mantener `SARIMAX -> GARCH` como opcion estable
- anadir una alternativa de modelo de media: `HAR-RV -> GARCH`
- enriquecer el set de senales de volatilidad con features dinamicas `sigma_*`
- mejorar robustez/velocidad del dataset builder con calculos vectorizados

El objetivo funcional es subir la capacidad del chain para capturar transiciones de regimen sin introducir leakage.

## 1) Cambios funcionales principales

Se implementaron 4 bloques:

1. soporte de `chain_mean_model` con dos modos: `sarimax` y `har`
2. nuevo modulo econometrico HAR-RV con fit y generacion de features
3. nuevas features de dinamica de volatilidad (`sigma_delta`, `sigma_ratio`, `sigma_diff1`, `sigma_diff2`, `abs_std_resid`)
4. feature `regime_boundary_distance` calculada de forma vectorizada

## 2) Integracion en `DatasetConfig` y pipeline

En [make_dataset.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/datasets/make_dataset.py) se consolido:

- `chain_mean_model: str = "sarimax"` con valores permitidos `sarimax|har`
- parametros HAR:
  - `har_target_col`
  - `har_lag_1`
  - `har_lag_week`
  - `har_lag_month`
  - `har_exog_cols`

La ruta `use_sarimax_garch_chain=True` ahora decide el primer bloque segun `chain_mean_model`:

- `sarimax` -> residuos de `sarimax_resid` para GARCH
- `har` -> residuos de `har_resid` para GARCH

## 3) Nuevo modulo HAR-RV

Se implemento [har_rv.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/econometric/har_rv.py):

- `HarRvConfig`
- `fit_har_rv(...)`
- `make_har_rv_features(...)`

Features generadas por HAR:

- `har_fcst_mean_h{h}`
- `har_fcst_se_h{h}`
- `har_ci_width_h{h}`
- `har_resid`
- `har_resid_std`

Detalles relevantes de implementacion:

- ajuste OLS por ticker
- estimacion de escala residual con `sqrt(SSE/(n-k))`
- validacion explicita de columnas exogenas faltantes
- en forecast multi-step con exogenas: **carry-forward de exog en t** para evitar look-ahead

## 4) Guardarrailes anti-leakage

Se reforzaron restricciones en dataset:

- prohibido `har_target_col="vol_fwd"`
- `use_sarimax_garch_chain=True` incompatible con `use_sarimax` o `use_garch` simultaneos
- validacion de columnas futuras prohibidas en exogenas (`vol_fwd`)
- validacion de `chain_mean_model` solo en `{"sarimax","har"}`

Esto mantiene el mismo rigor temporal que ya tenia la chain original.

## 5) Nuevas features dinamicas `sigma_*`

Tras generar el bloque de volatilidad (GARCH/EGARCH/GJR), se anadieron features derivadas:

- `sigma_t`
- `sigma_fwd_h{h}`
- `sigma_delta_h{h} = sigma_fwd_h{h} - sigma_t`
- `sigma_ratio_h{h} = sigma_fwd_h{h} / sigma_t`
- `sigma_diff1 = diff_1(sigma_t)`
- `sigma_diff2 = diff_1(sigma_diff1)` (aceleracion)
- `abs_std_resid = abs(z)` (cuando existe `*_z`)

Ademas se conservan prefijos por familia (`garch_*`, `egarch_*`, `gjrgarch_*`) y aliases genericos para que el tabular no dependa de la familia concreta.

## 6) `regime_boundary_distance` (vectorizada)

Se implemento/actualizo [\_regime_boundary_distance](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/datasets/make_dataset.py) para medir distancia de la senal de volatilidad a los cortes de regimen por ticker.

Puntos clave:

- calculo vectorizado con `reindex` por ticker + `numpy`
- evita `apply(axis=1)` costoso
- robusto a NaNs/cortes faltantes
- se anade como feature final: `regime_boundary_distance`

## 7) Integracion con variantes de volatilidad

En `make_dataset` se normalizo el dispatch de familia:

- `Garch`
- `EGARCH`
- `GJR-GARCH`

y se aplica el mismo bloque de features dinamicas `sigma_*` en los tres casos.

Esto deja una interfaz uniforme para el tabular aunque cambie la familia de volatilidad.

## 8) Integracion en walk-forward

En [walk_forward_chain_tab.py](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py):

- soporte completo de `chain_mean_model` en configs
- soporte de parametros HAR en perfiles/variantes
- validacion temprana de `profiles.full_axes.chain_mean_model`

Con esto se puede correr A/B limpio `sarimax vs har` dentro del mismo framework de seleccion y reporte.

## 9) Tests y validacion realizados

Tests de soporte HAR/chain:

- [test_har_rv_smoke.py](/home/manidmt/TFM/quant-risk-tfm/tests/test_har_rv_smoke.py)
- [test_chain_sarimax_garch.py](/home/manidmt/TFM/quant-risk-tfm/tests/test_chain_sarimax_garch.py)

Ejecucion verificada:

```bash
poetry run pytest -q tests/test_har_rv_smoke.py tests/test_chain_sarimax_garch.py
```

Resultado observado: `5 passed` (con warning de escala de `arch`, no bloqueante).

## 10) Compatibilidad hacia atras

No se rompio el modo anterior:

- si `chain_mean_model="sarimax"` el comportamiento sigue la logica previa
- HAR es opt-in por configuracion
- los artefactos de evaluacion (`fold_metrics.csv`, `summary.csv`, `final_vs_persistence.csv`) se mantienen

## 11) Impacto esperado en performance

La motivacion tecnica de esta feature:

- HAR suele ser competitivo para dinamica de volatilidad realizada
- `sigma_diff1/sigma_diff2/sigma_delta/sigma_ratio` aportan informacion de aceleracion y cambio de regimen
- `regime_boundary_distance` aporta senal de proximidad a cambio de clase

No garantiza mejora en todos los activos/horizontes, pero si amplia la capacidad del chain y el espacio de busqueda de forma consistente.

## 12) Siguientes pasos naturales

1. A/B formal `sarimax vs har` por activo/horizonte usando el mismo walk-forward.
2. Seleccionar por `macro_f1` + estabilidad + delta vs persistence.
3. Combinar con la capa `persistence-aware gating + calibration` para explotar mejor senales de transicion.
