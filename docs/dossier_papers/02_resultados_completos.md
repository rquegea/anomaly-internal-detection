# 02 — Resultados Completos: Tabla Maestra de Métricas

> Dataset primario: OPS-SAT-AD, cohort sampling=5, salvo indicación expresa.
> Protocolo anti-snooping: umbral fijado en validation/OOF, test evaluado ONE-SHOT.
> Generado 2026-04-13.

---

## A. Tabla Maestra — Todos los modelos

| Modelo | F0.5 | AUC | Precision | Recall | F1 | Val F0.5 | CI95 F0.5 | Features | Params | Script |
|---|---|---|---|---|---|---|---|---|---|---|
| IsolationForest | 0.381 | 0.701 | 0.367 | 0.451 | 0.405 | — | — | 18 | sklearn | s1_baselines/ |
| OneClassSVM | 0.669 | 0.800 | 0.656 | 0.726 | 0.689 | — | — | 18 | sklearn (RBF) | s1_baselines/ |
| Transformer v2 (p95) | 0.571 | 0.791 | 0.667 | 0.364 | 0.471 | — | — | windows 64pt | 71k | s2_transformer/ |
| Transformer v2 (sweep, p85) | 0.345 | 0.779 | 0.316 | 0.545 | 0.400 | 0.683 | — | windows 64pt | 71k | run_threshold_sweep.py |
| Transformer v2 (rebalanced) | 0.641 | 0.766 | — | — | — | — | — | windows 64pt | 71k | run_transformer_v2.py |
| LPI v1 (con n_peaks, full) | 0.801 | 0.978 | 0.862 | 0.625 | 0.725 | — | — | 18 | GMM K=15 | run_lpi_opssat.py |
| **LPI v1 (sin n_peaks) [claim v1]** | **0.670** | **0.920** | — | — | — | 0.646 | — | 17 | GMM K=15 | run_lpi_opssat.py |
| LPI normalizado (rate feats) | 0.446 | 0.896 | 0.556 | 0.250 | — | — | — | 17 norm | GMM K=15 | audit |
| LPINormalizingFlow (single seed=42) | 0.870 | 0.981 | 0.889 | 0.800 | — | 0.695 | — | 17 | 42564+2565 | compare_extensions.py |
| LPIVariational | 0.778 | 0.932 | — | — | — | — | — | 17 | DP-GMM K_eff=7 | compare_extensions.py |
| LPIBayesian | 0.670 | 0.920 | 0.833 | 0.375 | — | 0.646 | — | 17 | GMM K=15 (B=50) | compare_extensions.py |
| LPIHierarchical | 0.156 | 0.587 | 0.182 | 0.100 | — | 0.519 | — | 17 | 2-nivel K_m=6 | compare_extensions.py |
| LPIOnline | 0.129 | 0.504 | 0.115 | 0.250 | — | 0.253 | — | 17 | Online EM | compare_extensions.py |
| Ensemble NF+v1 (avg) | 0.784 | 0.976 | 0.780 | 0.800 | — | — | — | 16 | — | compare_extensions.py |
| NF ensemble mean (5 seeds) | 0.882 | 0.998 | — | — | — | 0.888 | [0.800, 0.936] | 16 | — | run_nf_seed_ensemble.py |
| **NF ensemble median (5 seeds) [CLAIM FINAL]** | **0.871** | **0.997** | — | — | — | **0.905** | **[0.780, 0.931]** | **16** | — | run_nf_seed_ensemble.py |
| NF ensemble rank (5 seeds) | 0.957 | 0.999 | — | — | — | 0.886 | [0.903, 0.994] | 16 | — | run_nf_seed_ensemble.py |

---

## B. Detalle por modelo — hiperparámetros completos

### B.1 IsolationForest

| Parámetro | Valor |
|---|---|
| n_estimators | 100 |
| contamination | 'auto' |
| random_state | 42 |
| Features | 18 estadísticas originales |
| Scaler | StandardScaler |
| Tiempo entrenamiento | [PENDIENTE: ejecutar run_baselines.py --time] |
| Tiempo inferencia/muestra | [PENDIENTE] |

### B.2 OneClassSVM

| Parámetro | Valor |
|---|---|
| kernel | rbf |
| nu | 0.1 |
| gamma | 'scale' |
| Features | 18 estadísticas originales |
| Scaler | StandardScaler |
| Tiempo entrenamiento | [PENDIENTE] |
| Tiempo inferencia/muestra | [PENDIENTE] |

### B.3 Transformer v2 (encoder-only)

| Parámetro | Valor |
|---|---|
| d_model | 64 |
| n_heads | 4 |
| n_layers | 2 |
| d_ff | 128 |
| window_size | 64 |
| stride | 32 |
| sampling_rate_filter | 5 |
| epochs | 30 |
| optimizer | Adam, lr=1e-3 |
| batch_size | 64 |
| loss | MSE reconstrucción |
| n_params | ~71 000 |
| Tiempo entrenamiento | [PENDIENTE] |
| Seed | 42 |

### B.4 LPI v1 (LPIDetector, claim sin n_peaks)

| Parámetro | Valor |
|---|---|
| n_components_range | (2, 15) |
| n_bootstrap | 20 |
| boot_size | 80% del train |
| scaler | RobustScaler |
| GMM covariance_type | full |
| GMM n_init | 5 |
| GMM max_iter | 300 |
| GMM reg_covar | 1e-6 |
| K seleccionado | 15 (BIC bootstrap) |
| CV folds | 5 (GroupKFold por segmento) |
| Features | 17 (sin n_peaks) |
| random_state | 42 |
| n_params | ~K*17 medias + K*17*17 cov + K pesos ≈ 4 460 |
| Tiempo entrenamiento | ~5s CPU (bootstrap BIC) |
| Tiempo inferencia/muestra | < 1 ms |

### B.5 LPINormalizingFlow (arquitectura hero)

| Parámetro | Valor |
|---|---|
| n_flow_layers | 4 (coupling layers RealNVP) |
| flow_hidden | 64 |
| flow_lr | 1e-3 |
| n_epochs | 200 |
| flow_patience | 30 (early stopping) |
| Params flow | 42 564 |
| GMM K | 15 (BIC bootstrap, mismo protocolo v1) |
| Params GMM | 2 565 |
| Total params | ~45 129 |
| Scaler | RobustScaler |
| Features | 16 (sin n_peaks, sin gaps_squared) para ensemble |
| Tiempo entrenamiento | ~21s CPU (single seed) |
| Tiempo inferencia/muestra | [PENDIENTE: medir forward pass NF+GMM] |
| CV OOF | 5-fold, GroupKFold por segmento |

### B.6 LPIVariational

| Parámetro | Valor |
|---|---|
| k_max | 15 |
| weight_concentration_prior | 0.01 (DP prior) |
| K efectiva resultante | 7 |
| covariance_type | full |
| n_init | 1 (BayesianGMM) |
| Scaler | RobustScaler |
| Features | 17 (sin n_peaks) |
| n_params | ~K_eff*17 medias + covarianzas ≈ 1 900 |
| Tiempo entrenamiento | ~2s CPU |
| Tiempo inferencia/muestra | < 1 ms |

### B.7 LPIBayesian

| Parámetro | Valor |
|---|---|
| n_bootstrap_bayes | 50 |
| n_components_range | (2, 15) (hereda LPIDetector) |
| n_bootstrap BIC | 20 (hereda) |
| K base | 15 (BIC bootstrap) |
| Scaler | RobustScaler |
| Features | 17 (sin n_peaks) |
| Output | score + CI90 por muestra |
| Tiempo entrenamiento | ~20s CPU (50 boots) |
| Tiempo inferencia/muestra | < 1 ms |

### B.8 NF ensemble (median, 5 seeds)

| Parámetro | Valor |
|---|---|
| Seeds | [0, 1, 42, 123, 999] |
| Estrategia | min-max normalize [0,1] → median entre seeds |
| Threshold | percentil OOF del ensemble (sweep val) |
| Bootstrap CI95 | B=1000, master_seed=42 |
| Features | 16 auditadas (sin n_peaks, sin gaps_squared) |
| Tiempo total | ~21s × 5 seeds ≈ 105s CPU |

---

## C. Resultados por seed — NF ensemble

| Seed | Val F0.5 | Test F0.5 | Test AUC |
|---|---|---|---|
| 0 | 0.771 | 0.769 | 0.972 |
| 1 | 0.838 | 0.785 | 0.967 |
| 42 | 0.633 | 0.707 | 0.950 |
| 123 | 0.800 | 0.792 | 0.973 |
| 999 | 0.708 | 0.855 | 0.971 |
| mean ± std | — | 0.782 ± 0.048 | 0.967 ± 0.009 |

---

## D. Threshold sweep — Transformer v2 (Decisión 4)

| Percentil | Val F0.5 | Val AUC |
|---|---|---|
| 70 | 0.677 | 0.803 |
| 75 | 0.664 | 0.803 |
| 80 | 0.664 | 0.803 |
| **85** | **0.683** | **0.803** |
| 90 | 0.631 | 0.803 |
| 92 | 0.606 | 0.803 |
| 95 | 0.570 | 0.803 |

Mejor umbral en val: p85 (F0.5=0.683). En test: F0.5=0.345. Gap val→test = 0.338 (distribución shift).

---

## E. Ablation features — NF (notebook 04, audit V1)

| Feature eliminada | Delta F0.5 | Interpretación |
|---|---|---|
| diff2_var | -0.342 | Feature mas critica; ver audit V1 |
| gaps_squared | -0.224 | Dominante pero length-confounded; excluida |
| n_peaks | — | Excluida previamente (audit 03) |
| Resto | < 0.05 | No críticas |

---

## F. Comparación cross-domain — LPI v1 (Paper 1)

| Dataset | N total | N clase rara | Ratio rara | AUC (5-fold CV) | Std |
|---|---|---|---|---|---|
| CHIME/FRB Catalog 2 | 3 641 | 83 repeaters | 2.3% | 0.760 | ± 0.048 |
| eROSITA dwarf galaxies | 169 | 95 IMBHs | 56.2% | 0.708 | ± 0.103 |
| OPS-SAT-AD (sampling=5) | 1 330 segs | [PENDIENTE: contar anómalos train+test] | — | 0.920 (test AUC) | — |

Nota: CHIME y eROSITA son datasets astrofísicos donde LPI v1 fue validado y constituyen la base del **Paper 1**. OPS-SAT-AD es el dataset de telemetría satelital para el **Paper 3**.

---

## G. Correlación Spearman de scores (test set, LPI v2)

| Par | Correlación |
|---|---|
| LPI v1 ↔ LPIBayesian | 1.000 (idéntica geometría GMM) |
| LPI v1 ↔ LPINormalizingFlow | 0.582 (señales distintas — potencial ensemble real) |
| LPIOnline ↔ todos | −0.4 a −0.7 (señal invertida — modelo degradado) |

---

## H. Mejora relativa vs baselines — NF ensemble median

| Baseline | F0.5 baseline | F0.5 NF ensemble | Delta F0.5 | Delta relativo |
|---|---|---|---|---|
| OneClassSVM | 0.669 | 0.871 | +0.202 | +30.2% |
| LPI v1 (sin n_peaks) | 0.670 | 0.871 | +0.201 | +30.0% |
| Transformer rebalanced | 0.641 | 0.871 | +0.230 | +35.9% |

CI95 lower bound (0.780) > mejor baseline (0.669) → superioridad estadísticamente demostrable.

---

*Ver `03_metodologia_lpi_v1.md` para formulación matemática del LPI v1.*
*Ver `04_metodologia_nf_ensemble.md` para formulación del NF ensemble.*
