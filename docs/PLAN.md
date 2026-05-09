## Plan Senior ML/Quant (Ordenado y Ejecutable)

### Resumen
Objetivo: mejorar `chain` frente a `persistence` sin romper el core actual, incorporando exógenas GDELT sin leakage y manteniendo reproducibilidad.  
Estrategia: `main` estable, PRs pequeños por capability, evaluación walk-forward homogénea por activo y horizonte.

### 1. Orden de ejecución (con ramas y merges)

| Orden | Rama | Objetivo | Merge a `main` cuando |
|---|---|---|---|
| 0 | `feature/new_models` (actual) | Consolidar TabPFN + EGARCH + GJR-GARCH ya implementado | `pytest -q` verde + smoke walk-forward h=5 por activo |
| 1 | `feat/vol-skewt-support` | Añadir `dist="skewt"` a GARCH/EGARCH/GJR + variantes YAML | Tests econométricos + comparación rápida vs tstudent |
| 2 | `feat/chain-har-sigma-features` | Añadir HAR-RV opcional + features `sigma_*` y “distancia a frontera de régimen” | `walk_forward` corto mejora o no-regresión en macro F1 |
| 3 | `feat/persistence-aware-gating-calibration` | Gating por confianza + calibración probas + threshold tuning por clase | Mejora en valid robust score y test no peor que baseline |
| 4 | `feat/gdelt-ingest-daily` | Ingesta GDELT GKG diaria agregada a DuckDB | Tests de esquema, incremental y calidad de datos |
| 5 | `feat/gdelt-features-no-leakage` | Features GDELT (count/tone/rolling/shock) con alineación temporal segura | Tests de no-leakage + build_features reproducible |
| 6 | `feat/gdelt-walkforward-eval` | Integrar GDELT al walk-forward y comparación formal vs baselines | Informe comparativo final por activo/horizonte |
| 7 (opcional) | `chore/mlflow-light-tracking` | Tracking de runs con MLflow (sin migración total) | Logging estable de params/metrics/artifacts |
| 8 (no implementar ahora) | `rfc/kedro-migration` | Documento de decisión arquitectónica (ADR) | Decisión explícita de “no migrar ahora” |

## 2. Flujo Git recomendado (push/merge)
1. Cerrar primero PR de `feature/new_models` a `main` (foundation).
2. Todas las ramas nuevas nacen de `main` actualizado, nunca de otra feature branch.
3. PR por feature (tabla anterior), `squash merge` para historia limpia.
4. Reglas de merge:
- CI mínimo: `poetry run pytest -q`.
- Smoke run obligatorio: `walk_forward_chain_tab.py` h=5 para `^GSPC`, `BTC-USD`, `TLT`.
- No se mergea si empeora fuerte vs `persistence` en valid (`delta_macro_f1_vs_persistence < -0.01` promedio).
5. Convención de push:
- Commits atómicos por bloque funcional.
- Push frecuente (`git push -u origin <branch>`) al cerrar cada bloque técnico.
- Resultado de experimentos fuera de Git (`runs/` ya ignorado), resumen en PR description.

## 3. Cambios de interfaz/API/tipos (planificados)

### 3.1 Econométricos
- Extender distribución soportada en:
  - [garch.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/econometric/garch.py)
  - [egarch.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/econometric/egarch.py)
  - [gjrgarch.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/econometric/gjrgarch.py)
- Nuevo valor: `dist: skewt`.
- Actualizar variantes en:
  - [chain_variants.yaml](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/econometric/chain_variants.yaml)
  - `chain_variants_egarch.yaml`, `chain_variants_gjrgarch.yaml`.

### 3.2 Features chain
- En [make_dataset.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/datasets/make_dataset.py):
  - Mantener `regime` como target principal.
  - Añadir features `sigma_delta`, `sigma_ratio`, `sigma_diff1`, `sigma_diff2`, `abs_std_resid`, `regime_boundary_distance`.
  - Añadir opción HAR-RV como bloque econométrico alternativo (misma interfaz de salida de features).

### 3.3 GDELT
- Nueva ingesta: `scripts/ingest_gdelt.py` y módulo `src/quant_risk/data/gdelt.py`.
- Nuevo bloque en [config/datasources.yaml](/home/manidmt/TFM/quant-risk-tfm/config/datasources.yaml):
  - `gdelt.enabled`, `start`, `lookback_buffer_days`, `publication_lag_bdays=1`.
- Extensión de [config/features.yaml](/home/manidmt/TFM/quant-risk-tfm/config/features.yaml):
  - `news_features.windows: [3,10,20]`, toggles por feature.
- En [build.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py):
  - `news_count`, `tone_mean`, `tone_std`, `tone_neg_share`, `attention_shock_z`.
  - Rolling `sum/mean/std` para 3/10/20d.
  - `shift(1 BDay)` antes de rolling para no leakage.

### 3.4 Walk-forward y selección
- En [walk_forward_chain_tab.py](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py):
  - Gating dinámico contra persistence por umbral de confianza.
  - Calibración (Platt o Isotonic) en valid fold.
  - Threshold tuning por clase para Macro F1.
  - Reporte adicional: performance en transiciones de régimen vs no transiciones.

## 4. Validaciones y criterios de aceptación

### Tests obligatorios
- `pytest -q` completo.
- Smoke nuevos:
  - `test_skewt_smoke.py` (familias vol).
  - `test_gdelt_ingest_smoke.py`.
  - `test_gdelt_no_leakage_alignment.py`.
  - `test_chain_gating_calibration_smoke.py`.

### Evaluación experimental (misma metodología)
- Mismos splits temporales, activos y horizontes (5, 20).
- Comparación siempre contra `Persistence`.
- Métrica primaria: `test macro_f1` en régimen.
- Métricas secundarias: `weighted_f1`, `macro_recall`, `high_vol_recall`, transición/no transición.

### Criterio de “éxito”
- `mean delta_macro_f1_vs_persistence > 0` en test por horizonte.
- Win-rate vs persistence ≥ 2/3 activos por horizonte.
- Sin degradación severa de `high_vol_recall` (< -0.03).

## 5. Valoración realista Kedro/MLflow

### Kedro
- Recomendación honesta: **no migrar ahora**.
- Motivo: coste alto de refactor de un backend ya avanzado, riesgo de regressions, beneficio predictivo nulo para el TFM en esta fase.
- Acción recomendada: crear solo un ADR (`rfc/kedro-migration`) con pros/contras y dejar migración post-TFM.

### MLflow
- Recomendación: **sí, pero light y opcional**.
- Implementación mínima:
  - backend local (file store),
  - logging de params/metrics/artifacts desde scripts de walk-forward,
  - sin rediseño de pipelines.
- Beneficio: trazabilidad y comparación de runs sin cambiar arquitectura core.

## 6. Supuestos y defaults fijados
- Se mantiene framing principal `regime` (no delta mode).
- Modelado por activo (`per_ticker`) como estándar.
- GDELT se integra primero como señal diaria global (sin NLP), replicada por ticker tras alineación temporal.
- `publication_lag_bdays=1` por defecto para noticias.
- Merge strategy por defecto: `squash`.
- `main` protegida; nada de pushes directos a `main`.

