# 07 — Baselines: Implementados y Recomendados

> Lista de baselines existentes en el repo y baselines recomendados para el Paper 3.

---

## A. Baselines implementados

### A.1 IsolationForest

| Campo | Valor |
|---|---|
| Script | `experiments/s1_baselines/run_baselines.py` |
| Clase | `sklearn.ensemble.IsolationForest` |
| Dataset | OPS-SAT-AD, sampling=5, features 18 |
| F0.5 (test) | 0.381 |
| AUC (test) | 0.701 |
| Hiperparámetros | n_estimators=100, contamination='auto', random_state=42 |
| Modo | Unsupervised |
| Tracking | MLflow (experimento s2_lpi_opssat, como baseline) |

**Descripción:** Construye árboles de aislamiento aleatorios y mide el número de particiones necesarias para aislar un punto. Puntos fáciles de aislar (pocos cortes) = anómalos. No usa labels.

**Por qué está aquí:** Baseline estándar de detección de anomalías tabulares. Resultado pobre (F0.5=0.381) porque los segmentos normales y anómalos se solapan en el espacio de features de 18 dimensiones.

---

### A.2 OneClassSVM

| Campo | Valor |
|---|---|
| Script | `experiments/s1_baselines/run_baselines.py` |
| Clase | `sklearn.svm.OneClassSVM` |
| Dataset | OPS-SAT-AD, sampling=5, features 18 |
| F0.5 (test) | 0.669 |
| AUC (test) | 0.800 |
| Hiperparámetros | kernel='rbf', nu=0.1, gamma='scale' |
| Modo | Unsupervised (entrena solo sobre datos normales) |

**Descripción:** Aprende una frontera de hiperesferoide en espacio de kernel RBF que encierra la mayoría de los datos normales. Puntos fuera del hiperesferoide = anómalos.

**Rol en el paper:** Principal baseline a superar. OCSVM F0.5=0.669 es el umbral de comparación. NF ensemble supera en +0.202 puntos (+30.2%). CI95 lower (0.780) > OCSVM → superioridad estadísticamente demostrable.

---

### A.3 Transformer v2 (encoder-only, reconstrucción MSE)

| Campo | Valor |
|---|---|
| Scripts | `experiments/s2_transformer/run_transformer_v2.py`, `run_threshold_sweep.py` |
| Clase | `src/models/transformer_ad.TransformerAD` |
| Dataset | OPS-SAT-AD, sampling=5, ventanas 64pt, stride=32 |
| F0.5 (test, rebalanced) | 0.641 |
| AUC (test, rebalanced) | 0.766 |
| Hiperparámetros | d_model=64, n_heads=4, n_layers=2, d_ff=128, epochs=30 |
| n_params | ~71 000 |
| Modo | Unsupervised (entrenado solo sobre segmentos normales) |

**Descripción:** Transformer encoder-only entrenado para reconstruir ventanas temporales normales. En inferencia, el error de reconstrucción MSE es el score de anomalía. Score de segmento = max sobre ventanas.

**Limitaciones identificadas:**
- Val→test gap de 0.338 puntos (distribución shift entre tipos de anomalía train/test)
- Requiere Optuna para mejorar (decidido no implementar — ver Decisión 5)
- El rebalanceo (oversampling de anómalos en train) subió de F0.5=0.345 → 0.641

---

### A.4 LPI v1 (LPIDetector)

| Campo | Valor |
|---|---|
| Script | `experiments/s2_lpi/run_lpi_opssat.py` |
| Clase | `src/models/lpi.LPIDetector` |
| Dataset | OPS-SAT-AD, sampling=5, 17 features (sin n_peaks) |
| F0.5 (test) | 0.670 |
| AUC (test) | 0.920 |
| Hiperparámetros | K_range=(2,15), n_bootstrap=20, scaler='robust' |
| Modo | Semi-supervised |

---

### A.5 LPIVariational

| Campo | Valor |
|---|---|
| Script | `experiments/s2_lpi_v2/compare_extensions.py` |
| Clase | `src/models/lpi_v2.LPIVariational` |
| Dataset | OPS-SAT-AD, sampling=5, 17 features |
| F0.5 (test) | 0.778 |
| AUC (test) | 0.932 |
| Hiperparámetros | k_max=15, weight_concentration_prior=0.01, K_eff=7 |
| Modo | Semi-supervised |

**Rol en el paper:** Alternativa ligera al NF. Útil para dispositivos edge o cuando el tiempo de entrenamiento es restricción.

---

### A.6 LPIBayesian

| Campo | Valor |
|---|---|
| Script | `experiments/s2_lpi_v2/compare_extensions.py` |
| Clase | `src/models/lpi_v2.LPIBayesian` |
| Dataset | OPS-SAT-AD, sampling=5, 17 features |
| F0.5 (test) | 0.670 |
| AUC (test) | 0.920 |
| Modo | Semi-supervised + incertidumbre |

**Diferenciación:** Produce CI90 por muestra (score_with_uncertainty()). Misma F0.5 que LPI v1 pero añade trazabilidad operacional.

---

### A.7 LPINormalizingFlow ensemble (claim final)

Ver `04_metodologia_nf_ensemble.md` para descripción completa.

| Campo | Valor |
|---|---|
| Script | `experiments/s2_lpi_v2/run_nf_seed_ensemble.py` |
| F0.5 (test) | 0.871 [CI95: 0.780–0.931] |
| AUC (test) | 0.997 [CI95: 0.993–1.000] |

---

## B. Baselines recomendados para Paper 3 (no implementados)

### B.1 DAGMM — Deep Autoencoding Gaussian Mixture Model

| Campo | Valor |
|---|---|
| Paper | Zong et al. (2018). AAAI. |
| Relevancia | Método seminal de deep anomaly detection con GMM en espacio latente — comparación directa con nuestra arquitectura NF+GMM |
| Diferencia con nuestro modelo | Autoencoder (non-bijective) vs RealNVP (bijective) + GMM; entrenamiento joint vs separado |
| Esfuerzo implementación | Medio — 1-2 días. Código PyTorch disponible públicamente. Adaptar a feature dataset de 16 dims. |
| Referencia código | https://github.com/danieltan07/dagmm (revisar licencia antes de usar) |

---

### B.2 Deep SVDD — Deep Support Vector Data Description

| Campo | Valor |
|---|---|
| Paper | Ruff et al. (2018). ICML. |
| Relevancia | Aprendizaje de representación + detección de anomalías unsupervised. Alternativa deep a OneClassSVM. |
| Diferencia | Minimiza el radio de una hipersfera en espacio latente aprendido. No usa GMM. |
| Esfuerzo implementación | Medio — 1-2 días. Adaptar a datos tabulares (no imagen). |
| Referencia código | https://github.com/lukasruff/Deep-SVDD-PyTorch |

---

### B.3 Anomaly Transformer

| Campo | Valor |
|---|---|
| Paper | Xu et al. (2022). ICLR. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." |
| Relevancia | State-of-the-art en anomaly detection sobre series temporales. Si este método ya está en el benchmark OPS-SAT-AD de KP Labs, hay que incluirlo o explicar por qué no. |
| Diferencia | Usa "association discrepancy" entre self-attention patterns como score de anomalía. Requiere datos de series temporales (segments.csv), no features tabulares. |
| Esfuerzo implementación | Alto — 3-5 días. Requiere pipeline de series temporales, no feature pipeline. Arquitectura compleja. |
| Referencia código | https://github.com/thuml/Anomaly-Transformer |

---

### B.4 Baselines oficiales KP Labs (30 métodos del paper benchmark)

| Campo | Valor |
|---|---|
| Fuente | https://github.com/kplabs-pl/OPS-SAT-AD (código oficial del paper Nature Sci. Data) |
| Número de baselines | 30 supervisados y no supervisados |
| Licencia | [VERIFICAR: licencia del repo oficial antes de usar código] |
| Relevancia | **CRÍTICO** — el paper de Paper 3 debe compararse con estos 30 baselines para ser aceptado. Un reviewer de NeurIPS pedirá esta comparación. |
| Baselines incluidos (selección) | KMeans, kNN, LOF, COPOD, HBOS, MCD, PCA-based, IsolationForest (config benchmark), OneClassSVM (config benchmark), LSTM-AE, etc. |
| Esfuerzo | 1-2 días — el código ya existe. Solo hay que ejecutarlo sobre el mismo split (sampling=5) y registrar resultados en MLflow. |

**ACCIÓN CRÍTICA:** Clonar el repo oficial de KP Labs, ejecutar los 30 baselines sobre el cohort sampling=5 con el mismo train/test split, y añadir los resultados a la tabla de comparación del paper. Sin esto el paper no es aceptable.

---

### B.5 Gonzalez et al. 2025 — Transformer sobre OPS-SAT-AD

| Campo | Valor |
|---|---|
| Paper | Gonzalez et al. (2025). "Transformers for anomaly detection in satellite telemetry." Acta Astronautica. DOI: 10.1016/j.actaastro.2025.xxx (ver S0094576525006095) |
| Relevancia | Demuestran que Transformers superan LSTM sobre OPS-SAT-AD. Es la referencia directa más relevante y es de sept 2025. |
| Qué reportan | Transformer F0.5 y AUC sobre OPS-SAT-AD (números exactos: [PENDIENTE — extraer del paper]) |
| Comparación necesaria | Nuestro NF ensemble debe superar estos números para que el Paper 3 sea una contribución real. |
| Esfuerzo | 0 días — solo leer el paper y añadir sus números a la tabla de comparación. |

---

## C. Tabla resumen: esfuerzo de implementación

| Baseline | Implementado | Esfuerzo pendiente | Prioridad para Paper 3 |
|---|---|---|---|
| IsolationForest | ✅ | 0 | Media |
| OneClassSVM | ✅ | 0 | Alta (principal) |
| Transformer v2 | ✅ | 0 | Alta |
| LPI v1 | ✅ | 0 | Alta |
| LPI v2 variants | ✅ | 0 | Alta |
| KP Labs 30 baselines | ❌ | 1-2 días | **CRÍTICA** |
| Gonzalez 2025 (leer paper) | ❌ | 0.5 días | **CRÍTICA** |
| DAGMM | ❌ | 1-2 días | Alta |
| Deep SVDD | ❌ | 1-2 días | Media |
| Anomaly Transformer | ❌ | 3-5 días | Media |
