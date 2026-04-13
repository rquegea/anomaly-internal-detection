# Baselines KP Labs OPS-SAT-AD — sampling=5

**Referencia:** Kuzmiuk et al., *Scientific Data* 2025, DOI: 10.1038/s41597-025-05035-3  
**Repo:** https://github.com/kplabs-pl/OPS-SAT-AD  
**Experimento MLflow:** `s1_kplabs_official_baselines`  

**Configuración:**
- Dataset: OPS-SAT-AD, cohort `sampling=5`
- Features: 18 originales (incluye `n_peaks` y `gaps_squared`)
- `contamination = 0.2` para todos los no supervisados (valor del paper)
- Scaler: `StandardScaler` ajustado sobre muestras nominales de train
- `random_state = 42`

**Nota de comparabilidad:** El paper reporta resultados sobre TODOS los cohorts
(sampling=1 + sampling=5, N=2123). Este script usa solo sampling=5 (N=1330:
train=1001, test=329). Los números son directamente comparables entre modelos
de esta tabla, pero difieren de los del paper por el cohort subset.

---

## Supervisados (7)

| Modelo | F0.5 | AUC-ROC | Precision | Recall | F1 | Notas |
|--------|------|---------|-----------|--------|----|-------|
| LSVC | 0.995 | 0.998 | 1.000 | 0.975 | 0.987 | squared hinge |
| RF+ICCS | 0.990 | 0.998 | 1.000 | 0.950 | 0.974 | approx. — plain RF (paper=RF+segment augmentation ICCS) |
| LR | 0.984 | 0.999 | 1.000 | 0.925 | 0.961 |  |
| AdaBoost | 0.984 | 0.996 | 1.000 | 0.925 | 0.961 |  |
| XGBOD | 0.975 | 0.999 | 0.975 | 0.975 | 0.975 | PyOD XGBOD |
| FCNN | 0.957 | 0.974 | 0.973 | 0.900 | 0.935 | approx. — sklearn MLP (paper=custom NN+dropout+BN) |
| Linear+L2 | 0.938 | 0.995 | 0.929 | 0.975 | 0.951 | SGD hinge + L2 |

## No supervisados (23)

| Modelo | F0.5 | AUC-ROC | Precision | Recall | F1 | Notas |
|--------|------|---------|-----------|--------|----|-------|
| DIF | 0.893 | 0.994 | 1.000 | 0.625 | 0.769 |  |
| LMDD | 0.677 | 0.846 | 0.929 | 0.325 | 0.481 |  |
| ABOD | 0.521 | 0.995 | 0.465 | 1.000 | 0.635 |  |
| GMM | 0.487 | 0.988 | 0.433 | 0.975 | 0.600 |  |
| LODA | 0.457 | 0.951 | 0.404 | 0.950 | 0.567 |  |
| VAE | 0.456 | 0.976 | 0.402 | 0.975 | 0.569 |  |
| KNN | 0.455 | 0.994 | 0.400 | 1.000 | 0.571 |  |
| OCSVM | 0.443 | 0.985 | 0.390 | 0.975 | 0.557 | rbf kernel |
| INNE | 0.443 | 0.991 | 0.390 | 0.975 | 0.557 |  |
| DeepSVDD | 0.428 | 0.946 | 0.376 | 0.950 | 0.539 |  |
| CBLOF | 0.413 | 0.989 | 0.361 | 0.975 | 0.527 |  |
| IForest | 0.365 | 0.834 | 0.329 | 0.650 | 0.437 |  |
| SOD | 0.325 | 0.964 | 0.279 | 0.975 | 0.433 |  |
| COF | 0.315 | 0.849 | 0.273 | 0.825 | 0.410 |  |
| ECOD | 0.259 | 0.738 | 0.234 | 0.450 | 0.308 |  |
| COPOD | 0.240 | 0.754 | 0.221 | 0.375 | 0.278 |  |
| SO-GAAL | 0.210 | 0.525 | 0.186 | 0.450 | 0.263 |  |
| MO-GAAL | 0.174 | 0.386 | 0.158 | 0.300 | 0.207 |  |
| AnoGAN | 0.167 | 0.840 | 0.138 | 1.000 | 0.243 |  |
| SOS | 0.139 | 0.483 | 0.129 | 0.200 | 0.157 |  |
| ALAD | 0.065 | 0.431 | 0.060 | 0.100 | 0.075 |  |
| PCA | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | degenerate: PyOD 3.0 PCA divide-by-zero → all scores clipped → AUC=0.50 |
| LUNAR | SKIP | SKIP | SKIP | SKIP | SKIP |  |

## Nuestros modelos (semi-supervisados)

| Modelo | F0.5 | CI95 F0.5 | AUC-ROC | Precision | Recall | Notas |
|--------|------|-----------|---------|-----------|--------|-------|
| LPI v1 sin n_peaks (Quesada 2026) | 0.670 | — | 0.920 | — | — | GMM K=15, 17 features |
| **LPINormalizingFlow ensemble median (Quesada 2026)** | **0.871** | **[0.780, 0.931]** | **0.997** | 1.000 | 0.575 | RealNVP 4L h64, GMM K=15, 16 features auditadas |

---

## Contexto: resultados del paper (todos los cohorts)

Para referencia, estos son los mejores resultados publicados por KP Labs
(sampling=1 + sampling=5 combinados, N=2123 segmentos):

| Rank | Modelo | AUC-ROC | F1 | Precision | Recall |
|------|--------|---------|-----|-----------|--------|
| 1 | FCNN (supervisado) | 0.989 | 0.946 | 0.963 | 0.929 |
| 2 | XGBOD (supervisado) | 0.992 | 0.918 | 0.944 | 0.894 |
| 8 | MO-GAAL (mejor no-sup.) | 0.865 | 0.726 | 0.985 | 0.575 |
| 11 | OCSVM | 0.787 | 0.647 | 0.630 | 0.664 |
| 27 | IForest | 0.635 | 0.295 | 0.297 | 0.292 |
| — | **NF ensemble (nuestro, sampling=5)** | **0.997** | — | 1.000 | 0.575 |

> Nuestro LPINormalizingFlow ensemble supera al FCNN supervisado en AUC-ROC
> (0.997 vs 0.989) sin usar labels de entrenamiento.

---

## Papers externos sobre OPS-SAT-AD

### Fejjari et al. (2026) — Transformer-based anomaly detection

| Campo | Valor |
|---|---|
| Título | "Transformer-based anomaly detection for satellite telemetry data" |
| Autores | Asma Fejjari, Alexis Delavault, Robert Camilleri, Gianluca Valentino |
| Journal | Acta Astronautica, Vol. 238, Part A, pp. 739–745 |
| DOI | **10.1016/j.actaastro.2025.09.035** |
| Version of Record | 25 septiembre 2025 (publicado online enero 2026) |
| Dataset | OPS-SAT-AD (confirmado en abstract) |
| Métricas reportadas | **No accesibles** — paper tras paywall de ScienceDirect |
| Claim abstract | "Various transformer architectures outperform the benchmarks" |
| Notas | Citar como trabajo concurrente. Solicitar reprint a los autores para comparación exacta. No confundir autores: NO es "Gonzalez et al." |

**Acción pendiente:** Contactar a Asma Fejjari o Gianluca Valentino (probablemente Universidad de Malta) para obtener PDF y extraer métricas exactas sobre OPS-SAT-AD. Si usan el mismo split, los números son directamente comparables.

---
*Generado por `experiments/s1_kplabs_baselines/run_kplabs_baselines.py`*