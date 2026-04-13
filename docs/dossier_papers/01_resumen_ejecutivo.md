# 01 — Resumen Ejecutivo del Repositorio

> Generado 2026-04-13. Referencia: CLAUDE.md decisiones 1–9 + exploración completa del repo.

---

## 1. Estado actual del repositorio

### Estructura de carpetas

```
anomaly-internal-detection/
├── src/
│   ├── data/loader.py          # Windowing, feature loading, scaling
│   ├── models/lpi.py           # LPIDetector v1 (clase base)
│   ├── models/lpi_v2.py        # 5 extensiones LPI (NF, Variational, Bayesian, Hierarchical, Online)
│   ├── models/transformer_ad.py # Transformer encoder-only baseline
│   └── evaluation/metrics.py   # fbeta_score, AUC, threshold sweep helpers
├── experiments/
│   ├── s1_baselines/run_baselines.py       # IsolationForest + OCSVM
│   ├── s2_lpi/run_lpi_opssat.py            # LPI v1 pipeline completo
│   ├── s2_lpi_v2/compare_extensions.py     # Comparativa 6 variantes LPI v2
│   ├── s2_lpi_v2/run_nf_seed_ensemble.py   # Ensemble 5 seeds + CI95
│   └── s2_transformer/                     # Threshold sweep + retrain balanceado
├── tests/
│   ├── test_lpi.py             # Tests LPI v1
│   ├── test_lpi_v2.py          # 45 tests extensiones LPI v2
│   ├── test_nf_ensemble.py     # 8 tests ensemble NF
│   └── test_no_length_leakage.py # 17 tests anti-leakage ventanas
├── notebooks/
│   ├── 01_explore_segments.ipynb         # EDA dataset crudo
│   ├── 02_val_test_distribution_shift.ipynb
│   ├── 03_lpi_integrity_audit.ipynb      # Audit n_peaks
│   └── 04_nf_integrity_audit.ipynb       # Audit NF + V1-V5 validaciones
├── reference/
│   ├── data/dataset.csv    # Features pre-computadas por segmento
│   ├── data/segments.csv   # Series temporales crudas
│   └── modeling_examples.ipynb
└── docs/dossier_papers/    # Este dossier
```

### Tests que pasan

| Suite | Tests | Estado |
|---|---|---|
| test_lpi_v2.py | 45/45 | PASS |
| test_nf_ensemble.py | 8/8 | PASS |
| test_no_length_leakage.py | 17/17 | PASS |

### Tracking de experimentos

MLflow activo en `mlflow.db`. Experimentos registrados: `s2_lpi_opssat`, `s2_lpi_v2_compare`, `s2_nf_ensemble_v7`.

---

## 2. Las 9 decisiones de diseño (audit trail)

| # | Fecha | Decisión | Impacto |
|---|---|---|---|
| 1 | 2026-04-12 | Transformer en lugar de LSTM autoencoder | Evidencia empírica: Gonzalez et al. 2025 muestra Transformers > LSTM en OPS-SAT-AD |
| 2 | 2026-04-12 | Ventanas deslizantes (`make_sliding_windows`) en lugar de segmentos completos | Elimina length leakage (anómalos 3.4× más largos) y sampling rate mixing |
| 3 | 2026-04-12 | Evaluación a nivel de segmento con `max` de errores de ventana | Alineado con protocolo benchmark OPS-SAT-AD; `max` captura ventana crítica |
| 4 | 2026-04-12 | Optuna necesario para Transformer (val→test gap 0.338) | Brecha por distribución shift entre anomalías train/test |
| 5 | 2026-04-12 | Saltar Optuna y sampling=1; ir directamente al LPI | Gap Transformer vs OCSVM = 0.028 (ruido estadístico); LPI da diferenciación con paper propio |
| 6 | 2026-04-12 | LPI como modelo principal | LPI v1 F0.5=0.801, AUC=0.978 — supera todos los baselines de S1 |
| 7 | 2026-04-12 | Audit integridad: n_peaks inflada, claim recalibrado sin n_peaks | n_peaks AUC individual=0.932; al normalizar por longitud cae a 0.674 — señal MIX (real + amplificada). Claim honesto: F0.5=0.670, AUC=0.920 |
| 8 | 2026-04-13 | LPI v2: NormalizingFlow como modelo hero | NF single seed F0.5=0.870, AUC=0.981 vs v1 F0.5=0.670 (+0.200 F0.5) |
| 9 | 2026-04-13 | NF ensemble median 5 seeds + CI95 como claim final publicable | CI95 lower=0.780 > 0.75 supera todos los baselines con certeza estadística; elimina cherry-picking |

---

## 3. Modelos consolidados (publicables)

### Modelo 1 — LPI v1 (Quesada 2026, Paper 1)

| Métrica | Valor |
|---|---|
| F0.5 (test, OPS-SAT-AD) | 0.670 |
| AUC (test, OPS-SAT-AD) | 0.920 |
| AUC (CHIME/FRB, 5-fold CV) | 0.760 ± 0.048 |
| AUC (eROSITA dwarfs, 5-fold CV) | 0.708 ± 0.103 |
| Features | 17 (sin n_peaks) |
| Parámetros | GMM K=15 seleccionado por BIC bootstrap |

Script: `experiments/s2_lpi/run_lpi_opssat.py`
CI95: no calculado (claim de punto).

### Modelo 2 — LPINormalizingFlow single seed

| Métrica | Valor |
|---|---|
| F0.5 (test) | 0.870 |
| AUC (test) | 0.981 |
| Features | 17 (sin n_peaks) |
| Seed | 42 |

**No publicable tal cual** — alta varianza de seed (std=0.088, rango 0.676–0.870). Ver Decisión 9.

### Modelo 3 — LPINormalizingFlow ensemble median, 5 seeds [CLAIM FINAL]

| Métrica | Valor | CI95 |
|---|---|---|
| F0.5 (test) | 0.871 | [0.780, 0.931] |
| AUC (test) | 0.997 | [0.993, 1.000] |
| F0.5 (val OOF) | 0.905 | — |
| Features | 16 auditadas (sin n_peaks, sin gaps_squared) |
| Seeds | [0, 1, 42, 123, 999] |

Script: `experiments/s2_lpi_v2/run_nf_seed_ensemble.py`
**Publicable en NeurIPS ML4PS / MNRAS Letters.** CI95 lower bound 0.780 > todos los baselines.

### Modelo 4 — LPINormalizingFlow ensemble rank (upper bound)

| Métrica | Valor | CI95 |
|---|---|---|
| F0.5 (test) | 0.957 | [0.903, 0.994] |
| AUC (test) | 0.999 | — |

Se reporta como hallazgo adicional; el claim conservador usa median.

### Modelos de producción complementarios

| Modelo | F0.5 | AUC | Rol |
|---|---|---|---|
| LPIVariational | 0.778 | 0.932 | Alternativa ligera (edge/inferencia rápida) |
| LPIBayesian | 0.670 | 0.920 | CI90 operacional para trazabilidad |

---

## 4. Modelos descartados

### LPIHierarchical

- **Resultado:** F0.5=0.156, AUC=0.587
- **Razón:** K_macro=6 demasiado grueso para el dataset (1001 segmentos de train). Gap val→test masivo (0.519→0.156). Para mejorar requeriría K_macro grande + tuning que converge al LPI v1, sin ganancia diferencial.
- Script: `experiments/s2_lpi_v2/compare_extensions.py`

### LPIOnline

- **Resultado:** F0.5=0.129, AUC=0.504
- **Razón:** warm_start EM sobre batches pequeños degrada los parámetros GMM. Score inversamente correlacionado con todos los modelos (Spearman ≈ −0.4 a −0.7) — señal invertida. Implementación correcta requiere sufficient statistics explícitas (Cappé & Moulines 2009), no warm_start sklearn.
- Script: `experiments/s2_lpi_v2/compare_extensions.py`

---

## 5. Features descartadas por leakage

### n_peaks — descartada (Decisión 7)

- **Qué es:** Número de picos en la serie temporal del segmento.
- **AUC individual (raw):** 0.932 — la feature más potente del set original.
- **Problema:** Segmentos anómalos son 3.4× más largos → tienen más picos por construcción, no por física.
- **Test de normalización:** `n_peaks_per_point` (AUC=0.702) y `n_peaks_per_sec` (AUC=0.674) — señal cae a mediocridad al normalizar.
- **Veredicto:** Señal MIX. Existe señal física real (anomalías oscilan ~1.8× más por punto) pero amplificada por longitud. Excluida del claim publicable.
- **Reincorporación:** Posible si telemetría SpainSat NG tiene segmentos homogéneos en longitud. [VERIFICAR: Rodrigo confirma longitud típica de segmentos en SpainSat NG.]

### gaps_squared — descartada (Decisión 9)

- **Qué es:** Cuadrado de la suma de gaps temporales en el segmento.
- **Problema detectado en notebook 04:** Feature dominante en ablation del NF (ΔF0.5=−0.224 al excluir). Su dominancia es potencialmente length-confounded (segmentos más largos → más gaps → mayor gaps_squared).
- **Veredicto:** Posible length confounding indirecto. Excluida de las 16 features auditadas para el claim final.
- **Feature set final (16 features):** Las 18 originales del dataset.csv, menos n_peaks y gaps_squared.

---

*Ver `02_resultados_completos.md` para tablas de métricas exhaustivas.*
*Ver `06_audit_trail.md` para los hallazgos detallados de los notebooks 03 y 04.*
