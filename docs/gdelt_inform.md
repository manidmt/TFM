# Informe de Resultados GDELT vs Persistence

## Introducción al informe

Este informe resume el estado actual de los experimentos `chain` (econométrico + tabular) frente a `persistence`, con foco en la integración de señales externas GDELT.

Alcance del consolidado:

- Fuente principal: [`runs/analysis_best_vs_persistence_20260311_164901/all_best_vs_persistence_rows.csv`](/home/manidmt/TFM/quant-risk-tfm/runs/analysis_best_vs_persistence_20260311_164901/all_best_vs_persistence_rows.csv)
- Resumen por run/horizonte: [`runs/analysis_best_vs_persistence_20260311_164901/all_best_vs_persistence_grouped.csv`](/home/manidmt/TFM/quant-risk-tfm/runs/analysis_best_vs_persistence_20260311_164901/all_best_vs_persistence_grouped.csv)
- Cobertura: `106` filas de best-by-asset, `35` runs, activos `^GSPC/BTC-USD/TLT`, horizontes `h=5` y `h=20`.

Lectura rápida global:

- `h=5`: media `target_delta_macro_f1_vs_persistence = +0.0163`
- `h=20`: media `target_delta_macro_f1_vs_persistence = -0.0032`
- No hay ningún run completo (`3` activos) con mejora positiva en `Macro F1` para los 3 activos simultáneamente.

## Mejores resultados por arquitectura y fecha del run

### Mejores arquitecturas completas (comparables: 3 activos)

Métrica de ranking: `mean_target_delta_macro_f1` (media por activo frente a persistence).

| Arquitectura | Mejor run | Fecha run | Horizonte | mean ΔMacro F1 target | mean ΔAcc target | mean ΔHighVolRecall target |
|---|---|---|---:|---:|---:|---:|
| `SARIMAX+GARCH + TABPFN + DELTA` | `walk_forward_chain_tabpfn_v25_best_20260305_140444` | 2026-03-05 14:04:44 | 5 | +0.0817 | +0.0078 | +0.0956 |
| `SARIMAX+GARCH + XGB + DELTA` | `walk_forward_chain_xgb_delta_best_20260304_212445` | 2026-03-04 21:24:45 | 5 | +0.0734 | -0.0032 | +0.0655 |
| `SARIMAX+GARCH + XGB + REGIME` | `walk_forward_chain_long_20260306_163601` | 2026-03-06 16:36:01 | 5 | +0.0114 | +0.0071 | +0.0064 |
| `SARIMAX+GARCH + TABPFN + REGIME` | `walk_forward_chain_tabpfn_v25_best_20260305_140444` | 2026-03-05 14:04:44 | 5 | +0.0031 | ~0.0000 | +0.0334 |

### Arquitecturas asimétricas / HAR (estado actual)

Estas arquitecturas existen, pero en la práctica están evaluadas sobre runs parciales/smoke (habitualmente 1 activo), por lo que no son comparables 1:1 con runs completos de 3 activos:

- `SARIMAX+EGARCH + TABPFN + REGIME`
- `SARIMAX+GJRGARCH + XGB + REGIME`
- `HAR+GARCH + TABPFN/XGB + REGIME`

La mejor fila parcial observada para asimétricos:

- `SARIMAX+GJRGARCH + XGB + REGIME` en `_smoke_asym_20260306_234833` (`h=5`, 1 activo) con `target_delta_macro_f1 ≈ +0.0257`.

## Porqué GDELT no está funcionando y bloques de código + lógica completa

### 1) Señal GDELT muy colineal entre temáticas

Con la ingesta actual, las series por `query_id` están casi superpuestas en dinámica:

- Correlaciones `news_count` entre `macro_us/crypto/rates` en `gdelt_gkg_daily`: ~`0.9998`.
- En features, `attn_z_macro_us`, `attn_z_crypto`, `attn_z_rates` aparecen prácticamente idénticas (correlación ~`1.000`).

Diagnóstico: aunque hay 3 temáticas, el modelo recibe señales casi redundantes; eso reduce la capacidad de aportar edge adicional sobre persistence.

### 2) La lógica de ingesta prioriza cobertura, no necesariamente diferenciación temática

Código relevante:

- Config y retries/backoff: [`src/quant_risk/data/gdelt.py:31`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gdelt.py#L31)
- Timeline por chunks (`TimelineVolRaw` + `TimelineTone`): [`src/quant_risk/data/gdelt.py:618`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gdelt.py#L618)
- Selección de candidatos de query por “viva/no viva”: [`src/quant_risk/data/gdelt.py:772`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gdelt.py#L772)

La heurística de “query viva” evita series muertas, pero no optimiza explícitamente independencia entre `query_id`s.

### 3) En feature engineering hay lag correcto anti-leakage, pero también fuerte suavizado

Código relevante:

- Carga y alineación de news block: [`src/quant_risk/features/build.py:89`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py#L89)
- Aplicación de `publication_lag_bdays` (shift): [`src/quant_risk/features/build.py:181`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py#L181)
- Dropeo de queries muertas: [`src/quant_risk/features/build.py:193`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py#L193)
- Interacciones news × vol: [`src/quant_risk/features/build.py:461`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py#L461)

Configuración actual en [`config/features.yaml:37`](/home/manidmt/TFM/quant-risk-tfm/config/features.yaml#L37):

- `compact_mode: true`
- `include_roll_sum/mean/std: false`
- `include_tone_std: false`
- `include_tone_neg_share: false`

Esto reduce dimensionalidad, pero también limita variedad de señal temporal de noticias.

### 4) El selector de modelos y blend/gating favorece soluciones conservadoras

Código relevante:

- Métricas de blend y deltas vs persistence: [`scripts/walk_forward_chain_tab.py:1080`](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py#L1080)
- Búsqueda de estrategia blend/gating: [`scripts/walk_forward_chain_tab.py:1211`](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py#L1211)
- Robust score de selección: [`scripts/walk_forward_chain_tab.py:2455`](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py#L2455)
- Evaluación final oficial en test: [`scripts/walk_forward_chain_tab.py:1739`](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py#L1739)

En varios runs recientes con GDELT, el blend acaba en parámetros cercanos a empate con persistence (ejemplo: `alpha` efectivos bajos o combinación que no desplaza la frontera de decisión), lo que estabiliza pero no mejora `Macro F1`.

### 5) Estado real de runs GDELT más recientes

Runs GDELT completos más relevantes:

- `walk_forward_chain_tab_gdelt_xgb_quick_20260310_104644`
  - `h5`: `mean_target_delta_macro_f1 ≈ +0.00025`
  - `h20`: `mean_target_delta_macro_f1 ≈ -0.07016`
- `wf_gdelt_postfix_xgb_20260311_071100`
  - `h5`: `mean_target_delta_macro_f1 ≈ -0.00136`
  - `h20`: parcial (`1` activo)
- `wf_gdelt_postfix_xgb_h20_light_20260311_083554`
  - `h20`: `mean_target_delta_macro_f1 = 0.0` (empate)

## Brainstorming de soluciones para mejorar a persistence

### Bloque A: Mejorar la señal GDELT (calidad informativa)

1. Rediseñar queries para bajar colinealidad entre `macro_us/crypto/rates`.
2. Añadir filtros semánticos y de contexto por query (por ejemplo, restricciones más específicas de mercado/rates/crypto).
3. Crear features diferenciales entre temas:
   - `news_count_macro_us - news_count_rates`
   - `attn_z_macro_us - attn_z_crypto`
   - `topic_share` relativo entre queries.
4. Incluir `tone_std` y `tone_neg_share` (ahora desactivadas) con ablation controlada.

### Bloque B: Mejorar la lógica de selección/modelado

1. Ejecutar ablation obligatoria por run:
   - con y sin news,
   - con y sin blend/gating,
   - para medir contribución marginal real.
2. Separar objetivo de selección para transiciones:
   - penalizar menos en no-transición y premiar más `delta_transition_macro_f1`.
3. Revisar rejilla de blend para evitar caer en empates triviales con persistence.
4. Incluir calibración/threshold tuning por activo con criterio explícito de ganancia vs persistence en valid.

### Bloque C: Protocolo experimental

1. Comparar solo runs homogéneos (3 activos, mismos folds/horizonte).
2. Evitar extrapolar conclusiones de smoke/1-activo para arquitectura global.
3. Añadir scoreboard formal por horizonte con:
   - `mean_target_delta_macro_f1`,
   - `win_rate` por activo,
   - `high_vol_recall_delta`.

## Aclaración sobre el potencial GDELT todavía no explotado al máximo

Tu afirmación es **correcta en gran parte**:

1. **Sí**, GDELT no se ha probado todavía con todo el potencial arquitectural.
   - Los runs GDELT cerrados y comparables que tenemos son principalmente `SARIMAX+GARCH + XGB + regime`.
2. **Sí**, no hay a día de hoy una batería completa GDELT con `TabPFN + EGARCH/GJRGARCH` comparable en 3 activos y dos horizontes.
3. **Sí**, parte de lo más reciente se ejecutó en modo “light” para cerrar resultados (`h20` con `max_struct=1`, `max_model=1`).
4. Adicionalmente, hubo un intento GDELT+TabPFN que no produjo resultados por error de alineación:
   - [`runs/walk_forward_chain_tab_gdelt_tabpfn_20260310_100759/logs/walk_h5.log`](/home/manidmt/TFM/quant-risk-tfm/runs/walk_forward_chain_tab_gdelt_tabpfn_20260310_100759/logs/walk_h5.log)
   - Error: `RuntimeError: Persistence alignment produced 0 valid rows.`

Conclusión ejecutiva:

- A día de hoy, con runs GDELT comparables disponibles, **no hay evidencia sólida de mejora consistente sobre persistence**.
- El siguiente salto no es “más de lo mismo”, sino mejorar la **independencia informativa de las señales GDELT** y ejecutar una matriz completa (`TabPFN`, `XGB`, `EGARCH/GJR`, no-light) bajo el mismo protocolo de evaluación.
