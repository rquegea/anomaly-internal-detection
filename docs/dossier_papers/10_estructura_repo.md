# 10 — Estructura del Repositorio

> Árbol completo con descripción de cada carpeta y archivo.
> Indica qué archivos son críticos para reproducir resultados y cuáles son working notes.

---

## Árbol completo

```
anomaly-internal-detection/
├── CLAUDE.md                          [CRÍTICO] Documentación principal: decisiones, resultados, roadmap
├── pyproject.toml                     Dependencias del proyecto (Python packages)
├── uv.lock                            Lock file de UV (reproducibilidad de entorno exacto)
├── mlflow.db                          Base de datos MLflow con historial de experimentos
├── .gitignore
│
├── src/                               Código fuente — modelos y utilities
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                  [CRÍTICO] Carga de datos, ventanas deslizantes, splits
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lpi.py                     [CRÍTICO] LPIDetector v1 — clase base
│   │   ├── lpi_v2.py                  [CRÍTICO] 5 extensiones LPI v2 (NF, Variational, Bayesian, Hierarchical, Online)
│   │   └── transformer_ad.py          Transformer encoder-only para anomaly detection
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py                 Métricas: fbeta_score, AUC, threshold sweep utilities
│
├── experiments/                       Scripts de experimentos — generan los resultados del paper
│   ├── s1_baselines/
│   │   └── run_baselines.py           [CRÍTICO] IsolationForest + OneClassSVM S1
│   ├── s2_lpi/
│   │   └── run_lpi_opssat.py          [CRÍTICO] LPI v1 pipeline completo con anti-snooping
│   ├── s2_lpi_v2/
│   │   ├── compare_extensions.py      [CRÍTICO] Comparativa 6 variantes LPI v2
│   │   └── run_nf_seed_ensemble.py    [CRÍTICO] NF ensemble 5 seeds + CI95 (CLAIM FINAL)
│   └── s2_transformer/
│       ├── run_threshold_sweep.py     Threshold sweep Transformer v2
│       ├── run_transformer_v2.py      Entrenamiento Transformer v2
│       └── run_transformer_smoke.py   Smoke test rápido del Transformer
│
├── tests/                             Suite de tests — verifican integridad del pipeline
│   ├── test_lpi.py                    Tests básicos LPI v1
│   ├── test_lpi_v2.py                 [CRÍTICO] 45 tests de las 5 extensiones LPI v2
│   ├── test_nf_ensemble.py            [CRÍTICO] 8 tests del NF ensemble
│   └── test_no_length_leakage.py      [CRÍTICO] 17 tests anti-leakage de ventanas deslizantes
│
├── notebooks/                         Análisis exploratorio y audits — working notes
│   ├── 01_explore_segments.ipynb      [REFERENCIA] EDA del dataset crudo (hallazgo leakage)
│   ├── 01_explore_segments_executed.ipynb  Copia ejecutada del anterior (con outputs)
│   ├── 02_val_test_distribution_shift.ipynb  Análisis del distribution shift val→test Transformer
│   ├── 03_lpi_integrity_audit.ipynb   [REFERENCIA] Audit n_peaks — Decisión 7
│   └── 04_nf_integrity_audit.ipynb    [REFERENCIA] Audit NF (V1-V5) — Decisión 8 y 9
│
├── reference/                         Dataset oficial y código de referencia KP Labs
│   ├── README.md                      Descripción del dataset OPS-SAT-AD
│   ├── LICENSE                        Licencia del código de referencia KP Labs
│   ├── data/
│   │   ├── README.md                  Instrucciones de descarga del dataset
│   │   ├── dataset.csv                [CRÍTICO] Features pre-computadas por segmento (18 features)
│   │   └── segments.csv               [CRÍTICO] Series temporales crudas (para windowing)
│   ├── dataset_generator.ipynb        Notebook original KP Labs para generar el dataset
│   └── modeling_examples.ipynb        Ejemplos de modelos de KP Labs (baselines oficiales)
│
├── data/                              Figuras del EDA generadas
│   └── processed/
│       ├── fig1_segment_lengths.png   Distribución de longitudes de segmentos
│       ├── fig2_channel_distribution.png
│       ├── fig3_duration_distribution.png
│       ├── fig4_series_examples.png   Ejemplos de series temporales normal vs anómala
│       ├── fig5_overlay_comparison.png
│       └── fig6_values_by_channel.png
│
├── plots/                             Figuras de audits y experimentos
│   ├── 02_p1_channel_distribution.png
│   ├── 02_p1b_recall_by_channel.png
│   ├── 02_p2_error_distributions.png
│   ├── 02_p3_visual_inspection.png
│   ├── 03_decisive_scatter.png        [REFERENCIA] Scatter n_peaks vs longitud (leakage evidence)
│   ├── 03_q1_channel_bias.png         Distribución de anomalías por canal
│   ├── 03_q2_feature_auc.png          AUC individual por feature
│   └── 03_q3_ablation.png             Ablation LPI con/sin n_peaks
│
└── docs/                              Documentación generada
    └── dossier_papers/                Este dossier — 10 ficheros
        ├── 01_resumen_ejecutivo.md
        ├── 02_resultados_completos.md
        ├── 03_metodologia_lpi_v1.md
        ├── 04_metodologia_nf_ensemble.md
        ├── 05_datasets.md
        ├── 06_audit_trail.md
        ├── 07_baselines_actuales.md
        ├── 08_papers_a_citar.md
        ├── 09_gaps_y_pendientes.md
        └── 10_estructura_repo.md      (este fichero)
```

---

## Archivos críticos para reproducir resultados

Para reproducir el claim final (Paper 3, NF ensemble):

```
# 1. Instalar entorno
uv sync  # o pip install -e .

# 2. Verificar datos
ls reference/data/dataset.csv reference/data/segments.csv

# 3. Ejecutar claim final
python experiments/s2_lpi_v2/run_nf_seed_ensemble.py
# Output esperado: F0.5=0.871, AUC=0.997, CI95=[0.780, 0.931]

# 4. Verificar tests
pytest tests/ -v
# Esperado: 45+8+17 passed, 0 failed
```

Para reproducir el claim LPI v1 (Paper 1):

```
python experiments/s2_lpi/run_lpi_opssat.py
# Output esperado: F0.5=0.670, AUC=0.920
```

---

## Descripción detallada de archivos clave

### `src/data/loader.py` — Cargador y preprocesador

**Funciones principales:**

| Función | Línea aprox. | Descripción |
|---|---|---|
| `load_opssat_features()` | ~30 | Carga dataset.csv y filtra por sampling rate y split |
| `make_sliding_windows()` | ~90 | Genera ventanas deslizantes con z-scoring por segmento |
| `segment_scores_from_windows()` | ~180 | Agrega scores de ventanas a nivel de segmento (max/mean) |
| `scale_for_unsupervised()` | ~220 | StandardScaler sobre muestras normales de train |

**Parámetros clave de `make_sliding_windows()`:**
- `window_size=64`, `stride=32`: configuración validada para sampling=5
- `sampling_rate_filter=5`: cohort principal; usar `=1` para Fase 2
- `split='train'/'test'/'both'`: control del split

### `src/models/lpi.py` — LPIDetector v1

**Clases y métodos:**

| Elemento | Descripción |
|---|---|
| `LPIDetector.__init__()` | Configura K_range, n_bootstrap, scaler, random_state |
| `LPIDetector._select_k_by_bic()` | BIC bootstrap — selecciona K* |
| `LPIDetector.fit()` | Scale → BIC K selection → GMM fit → enrichment computation |
| `LPIDetector.score()` | LPI(x) = responsibilities @ enrichments |
| `LPIDetector.predict()` | Binariza scores por umbral |
| `LPIDetector.fit_predict_cv()` | 5-fold CV para OOF scores (anti-snooping) |

### `src/models/lpi_v2.py` — Extensiones LPI v2

**Clases:**

| Clase | Hereda de | Innovación | Rol en paper |
|---|---|---|---|
| `LPINormalizingFlow` | `LPIDetector` | RealNVP bijective mapping X→Z antes de GMM | **Modelo principal Paper 3** |
| `LPIVariational` | `LPIDetector` | Dirichlet process prior; K_eff automático | Alternativa ligera |
| `LPIBayesian` | `LPIDetector` | Bootstrap CI90 por muestra | Trazabilidad operacional |
| `LPIHierarchical` | `LPIDetector` | GMM 2-nivel macro+micro | **Descartado** (F0.5=0.156) |
| `LPIOnline` | `LPIDetector` | Streaming EM updates | **Descartado** (F0.5=0.129) |

**Helper interno:**
- `_AffineCouple(nn.Module)`: coupling layer afín del RealNVP
- `_RealNVP(nn.Module)`: stack de coupling layers con log-likelihood
- `_train_flow()`: entrenamiento Adam + early stopping

### `experiments/s2_lpi_v2/run_nf_seed_ensemble.py` — Script del claim final

**Protocolo implementado:**
1. Loop sobre seeds=[0, 1, 42, 123, 999]
2. Por seed: 5-fold GroupKFold CV → OOF scores, fit final → test scores
3. normalize_minmax() por seed
4. Tres estrategias ensemble: mean, median, rank
5. Threshold sweep sobre OOF ensemble → tau*
6. ONE-SHOT test evaluation
7. Bootstrap CI95 (B=1000, master_seed=42)

### `tests/test_no_length_leakage.py` — Suite anti-leakage

**17 tests que verifican:**
- Todas las ventanas tienen shape exacta `(window_size,)`
- Distribución de longitudes idéntica entre clases 0 y 1
- No NaN/Inf en ventanas
- < 1% de ventanas con std=0 (edge case de z-scoring)
- Cohorts sampling=1 y sampling=5 sin intersección de segmentos

---

## Archivos que son solo working notes (no críticos)

| Archivo | Por qué no es crítico |
|---|---|
| `notebooks/01_explore_segments_executed.ipynb` | Duplicado del notebook 01 con outputs guardados; no añade información nueva |
| `notebooks/02_val_test_distribution_shift.ipynb` | Análisis exploratorio que informó Decisión 4; resultado ya documentado en CLAUDE.md |
| `experiments/s2_transformer/run_transformer_smoke.py` | Smoke test para verificar que el Transformer corre; no genera resultados |
| `data/processed/*.png` | Figuras EDA intermedias; las definitivas están en `plots/` |
| `mlflow.db` | Útil para consultar historial pero no necesario para reproducción |
