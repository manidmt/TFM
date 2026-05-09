# feature: gdelt data ingestion

## 0) Objetivo de esta feature

Añadir una ingesta diaria de señales de noticias (GDELT) y conectarla al pipeline de features sin introducir leakage temporal.

El alcance implementado en esta rama cubre:

- ingesta incremental en DuckDB (`gdelt_gkg_daily`)
- soporte multi-query temático (`query_id`)
- configuración en `datasources.yaml`
- generación de features de noticias en `build_features`
- tests de humo y alineación temporal

## 1) Cambios funcionales principales

- Nuevo módulo de datos:
  - `src/quant_risk/data/gdelt.py`
- Nuevo script de ingesta:
  - `scripts/ingest_gdelt.py`
- Configuración GDELT en:
  - `config/datasources.yaml`
- Configuración de features de noticias en:
  - `config/features.yaml`
- Integración en el builder:
  - `src/quant_risk/features/build.py`
  - `scripts/build_features.py`
- Tests añadidos:
  - `tests/test_gdelt_ingest_smoke.py`
  - `tests/test_gdelt_no_leakage_alignment.py`

## 2) Nuevo módulo `gdelt.py`

En `src/quant_risk/data/gdelt.py` se implementa:

- `GdeltIngestConfig`: parámetros de ingesta (query, ventana incremental, timeout, etc.).
- `init_schema`: crea/migra tabla `gdelt_gkg_daily` con PK por `(date, query_id)`.
- `fetch_timeline_volraw` y `fetch_timeline_tone`: descarga agregados diarios por rango con GDELT Doc API en modo timeline.
- `fetch_artlist_range`: mantiene soporte de muestreo `ArtList` para debug/fallback.
- `fetch_timeline_range_for_query`: mergea `news_count` + `tone_mean` por query temática.
- `_queries_from_cfg`: resuelve `gdelt.queries` a `{query_id: query_text}`.
- `aggregate_daily`: agrega muestras ArtList cuando se usa modo `artlist`.
- `upsert_gdelt_daily`: upsert idempotente por `(date, query_id)`.
- `refresh_gdelt`: flujo incremental completo por rango y por query temática con `mode=timeline|artlist`.

## 3) Script CLI `ingest_gdelt.py`

Nuevo script `scripts/ingest_gdelt.py`:

- lee config desde `config/datasources.yaml`
- permite overrides por CLI (`--db`, `--start`, `--end`, `--query`)
- respeta `gdelt.enabled`
- ejecuta `refresh_gdelt` y devuelve estado estructurado

Comando base:

```bash
PYTHONPATH=src poetry run python scripts/ingest_gdelt.py --config config/datasources.yaml
```

## 4) Configuración añadida

### `config/datasources.yaml`

Nuevo bloque:

- `gdelt.enabled`
- `gdelt.table`
- `gdelt.start`
- `gdelt.lookback_buffer_days`
- `gdelt.publication_lag_bdays`
- `gdelt.mode` (`timeline` por defecto)
- `gdelt.keep_artlist_sample`
- `gdelt.query`
- `gdelt.queries` (mapa de queries temáticas)
- `gdelt.max_records_per_day`
- `gdelt.timeout_seconds`

Default actual: `enabled: false` para no romper el flujo previo.
Default actual de ingesta cuando se activa: `mode: timeline`.

### `config/features.yaml`

Nuevo bloque:

- `news_features.enabled`
- `news_features.windows` (por defecto `[3, 10, 20]`)
- `news_features.include_roll_sum`
- `news_features.include_roll_mean`
- `news_features.include_roll_std`
- `news_features.include_tone_std`
- `news_features.include_tone_neg_share`

Default actual: `enabled: false`.

## 5) Integración en `build_features` (no-leakage)

En `src/quant_risk/features/build.py`:

- se amplía `BuildFeaturesConfig` con parámetros GDELT/news.
- se añade `news_query_ids` para pivot por temática.
- se añaden toggles para controlar explícitamente si `tone_std` y `tone_neg_share` entran al set de features.
- se añade `_load_news_block(...)`:
  - carga `gdelt_gkg_daily`
  - alinea por `master_idx`
  - aplica `publication_lag_bdays` antes de cualquier rolling
  - calcula `attention_shock_z` sobre `news_count`
  - calcula rolling `sum/mean/std` para ventanas configuradas
- cuando hay `query_id`, pivota a columnas por tema (`news_count_macro_us`, `tone_mean_crypto`, etc.).
- si los toggles están desactivados, no se generan columnas raw/rolling de `tone_std` y `tone_neg_share`.
- el bloque de noticias se mergea por `date` en cada ticker.
- se ajusta `min_lag` para incluir lag + ventana de noticias cuando están activadas.

En `scripts/build_features.py`:

- se cablean los nuevos campos de `datasources.yaml` y `features.yaml` hacia `BuildFeaturesConfig`.
- se pasan `news_query_ids` desde `gdelt.queries`.

## 6) Guardarraíles anti-leakage aplicados

- `publication_lag_bdays` se aplica antes del cálculo rolling.
- las ventanas rolling usan solo histórico disponible tras ese shift.
- los tests validan explícitamente que `news_count` observado en `features_daily` equivale a `news_count.shift(1)` cuando el lag es 1.

## 7) Tests y validación ejecutada

Tests nuevos:

- `tests/test_gdelt_ingest_smoke.py`
- `tests/test_gdelt_no_leakage_alignment.py`

Ejecuciones realizadas:

```bash
PYTHONPATH=src poetry run pytest -q tests/test_gdelt_ingest_smoke.py tests/test_gdelt_no_leakage_alignment.py tests/test_build_features.py
PYTHONPATH=src poetry run pytest -q
```

Resultado final:

- `78 passed, 1 skipped`

## 8) Compatibilidad hacia atrás

No se rompe el pipeline actual:

- si `gdelt.enabled=false`, la ingesta se omite.
- si `news_features.enabled=false`, no se añaden columnas news al dataset.
- modo `artlist` sigue disponible para debug/fallback.
- defaults conservadores mantienen comportamiento legacy.
- si la tabla antigua no tenía `query_id`, se migra automáticamente manteniendo datos como `query_id='default'`.

## 9) Uso recomendado para activar GDELT

1. Poner `gdelt.enabled: true` en `config/datasources.yaml`.
2. Poner `news_features.enabled: true` en `config/features.yaml`.
3. Ejecutar:
   - `scripts/ingest_gdelt.py`
   - `scripts/build_features.py`
4. Lanzar los entrenamientos / walk-forward habituales.

## 10) Siguientes pasos naturales

- Integrar estas features en runs walk-forward comparativos por activo/horizonte.
- Medir delta frente a persistence con y sin news.
- Si la señal aporta, abrir siguiente rama para `feat/gdelt-walkforward-eval`.
