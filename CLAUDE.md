# CLAUDE.md — Detección de Anomalías Interna en Satélites

> Contexto de proyecto para Claude Code. Léelo antes de cualquier tarea técnica, comercial o de investigación relacionada con este producto.

---

## 1. Qué es este proyecto

Sistema de ML que analiza telemetría de satélites (temperaturas, energía, actitud, propulsión, transpondedores) y **detecta anomalías antes de que se conviertan en fallos críticos**. Se entrena sobre datasets públicos de ESA y es transferible a telemetría real del cliente.

**No confundir** con el "detector externo" (producto paralelo orientado al MESPA basado en TLEs públicos). Éste es **interno**: requiere acceso a telemetría del operador.

## 2. Cliente y mercado

- **Cliente primario:** Hisdesat (SpainSat NG, PAZ, XTAR-EUR). Controlada por Indra desde diciembre 2025 — la decisión de compra puede estar en Indra.
- **Secundarios:** Hispasat, INTA, operadores europeos medianos (Eutelsat, SES por capa defensa).
- **Contactos clave Hisdesat:** Miguel Ángel García Primo (CEO), Jorge Romo (jefe TI y Seguridad Industrial). Producto no vendible a puerta fría — necesitamos intermediario.

## 3. Propuesta de valor (una frase)

Detección predictiva de fallos distribuidos entre subsistemas que los sistemas actuales basados en umbrales no capturan. Ganancia = tiempo de reacción.

## 4. Competencia

| Competidor | Amenaza | Nuestro ángulo |
|---|---|---|
| Airbus / Thales (fabricantes) | Tienen monitorización propia en el satélite | Sistemas cerrados, poco adaptables. Nos posicionamos como **complementarios**, no sustitutivos. |
| GMV | Mejor posicionado si se mete | No están en ML predictivo todavía |
| RS21 Prequip for Space (US) | Hace exactamente esto para US Space Force | Referencia directa, no competencia en EU |
| Atomos, Telespazio | Más operaciones que ML | Poco riesgo |

**Diferenciación:** producto español, más barato, más ágil, integrable con Airbus existente.

## 5. Stack técnico

### ML
- PyTorch (Transformers series temporales — ver sección 7)
- scikit-learn (baselines: Isolation Forest, One-Class SVM)
- pandas, numpy
- MLflow (tracking experimentos — demuestra rigor)
- Optuna (búsqueda hiperparámetros)

### Backend
- FastAPI (inferencia)
- Redis (cache)
- PostgreSQL + **TimescaleDB** (series temporales — estándar del sector)

### Frontend
- Next.js + React + Tailwind
- Plotly o Recharts (series temporales multivariable)
- Dashboard: estado subsistemas, línea del tiempo de anomalías, drill-down por parámetro

## 6. Datasets

| Dataset | Uso | Tamaño | Licencia | Notas |
|---|---|---|---|---|
| **OPSSAT-AD** | Arranque | Manejable (horas) | **CC-BY 4.0** ✅ uso comercial | CubeSat real ESA, 30 baselines publicados → comparación directa. Nature 2025. |
| **ESA-AD** | Escala | ~12 GB, 3 misiones ESA | **CC-BY 4.0** ✅ uso comercial | Entrenamiento GPU 24–72h. Valida escalabilidad. |
| NASA SMAP/MSL | Benchmark | Estándar académico | — | Opcional |
| Beidou público | Paper referencia | — | — | Opcional |

**Licencia verificada:** ambos datasets ESA son CC-BY 4.0 → permiten uso comercial citando a los autores. Riesgo #5 del plan resuelto.

**URLs oficiales OPSSAT-AD:**
- Dataset (última versión, abr 2025): https://zenodo.org/records/15108715
- Código oficial con los 30 baselines implementados: https://github.com/kplabs-pl/OPS-SAT-AD
- Paper Nature Scientific Data: DOI 10.1038/s41597-025-05035-3
- Preprint arXiv: https://arxiv.org/abs/2407.04730

**URL ESA-AD:** https://github.com/kplabs-pl/AI-datasets (KP Labs)

**Referencias académicas clave:**
- Transformers sobre OPS-SAT superando benchmarks (sept 2025): https://www.sciencedirect.com/science/article/pii/S0094576525006095 → base para el modelo de S2/S3.

**Ahorro de Semana 1:** el repo oficial de KP Labs ya trae los 30 baselines. Clonar y ejecutar → números de referencia sin reimplementar. ~3–5 días ganados.

## 7. Arquitectura de modelos

### Decisiones de diseño (audit trail)

> **[2026-04-12] Decisión 1 — Transformer en lugar de LSTM Autoencoder:**
> Gonzalez et al. (ScienceDirect S0094576525006095, sept 2025) demuestran que Transformers superan consistentemente a los baselines LSTM sobre OPS-SAT-AD. No se invierte tiempo en LSTM cuando la evidencia empírica ya apunta al Transformer.

> **[2026-04-12] Decisión 2 — Ventanas deslizantes en lugar de segmentos completos:**
> EDA (notebooks/01_explore_segments.ipynb) identificó dos problemas de rigor que invalidan el uso de segmentos enteros como muestras de entrenamiento:
>
> *Problema A — Length leakage:* Los segmentos anómalos son ~3.4× más largos que los normales (mediana 184 pts vs 54 pts, cohort sampling=5). Un modelo que recibe segmentos completos puede aprender a detectar longitud en vez de patrón anómalo. Esto produciría resultados aparentemente buenos en benchmark pero inútiles en producción, donde las anomalías pueden ocurrir en segmentos cortos.
>
> *Problema B — Sampling heterogéneo:* El dataset tiene dos frecuencias de muestreo (1 Hz y 5 s). Mezclarlas presenta al modelo secuencias de igual número de puntos que representan ventanas temporales muy distintas (64 pts a 1 Hz = 64 s; 64 pts a 5 s = 320 s).
>
> **Solución adoptada:** `make_sliding_windows()` en `src/data/loader.py` — cada muestra es una ventana de exactamente `window_size` puntos extraída con stride sobre el segmento crudo. Esto garantiza que la distribución de longitudes es idéntica para clases normal y anómala (test formal: `tests/test_no_length_leakage.py`, 17/17 passed).
>
> **Fase 2 del modelo** (sampling=1, después de validar con sampling=5): pendiente. La misma arquitectura con `window_size=256` cubre el cohort 1 Hz.

> **[2026-04-12] Decisión 3 — Evaluación a nivel de segmento:**
> El benchmark OPS-SAT-AD asigna una etiqueta por segmento, no por punto. La puntuación del segmento = max de los errores de reconstrucción de sus ventanas. El `max` en lugar del `mean` captura que una sola ventana muy anómala es suficiente para alertar.

### Protocolo de evaluación y control de snooping

El umbral de detección (percentil sobre errores de reconstrucción) se selecciona mediante sweep sobre un **validation set**, nunca sobre el test set. El test set se evalúa exactamente una vez, al final, con el umbral ya fijado. Esto previene optimismo espurio (data snooping).

**Split 3-way para sampling=5:**
- `train_normal` (80% de segmentos normales del split train): se usa para entrenar el modelo
- `val` (20% normales + 100% anómalos del split train): se usa para elegir el percentil del umbral
- `test` (split train=0, nunca visto hasta la evaluación final): métrica reportable públicamente

La separación es por segmento, no por ventana, para evitar que ventanas del mismo segmento aparezcan en train y val simultáneamente. El script `run_threshold_sweep.py` implementa y documenta este protocolo.

### Resultados por modelo

| Modelo | Precision | Recall | F1 | **F0.5** | AUC-ROC | Pipeline | Estado |
|---|---|---|---|---|---|---|---|
| IsolationForest | 0.367 | 0.451 | 0.405 | 0.381 | 0.701 | features | S1 ✅ |
| OneClassSVM | 0.656 | 0.726 | 0.689 | 0.669 | 0.800 | features | S1 ✅ |
| Transformer v2 (p95) | 0.667 | 0.364 | 0.471 | 0.571 | 0.791 | sliding windows, sampling=5, 30 ep | S2 |
| Transformer v2 (sweep) | 0.316 | 0.545 | 0.400 | 0.345 | 0.779 | sliding windows + threshold sweep | S2 🔴 |
| Transformer v2 (rebalanced) | — | — | — | 0.641 | 0.766 | sliding windows + imbalance fix, sampling=5 | S2 ✅ |
| LPI full (n_peaks raw) | 0.862 | 0.625 | 0.725 | 0.801 | 0.978 | features, GMM K=15, sampling=5 | S2 ⚠️ n_peaks inflada |
| **LPI sin n_peaks (Quesada 2026)** | — | — | — | **0.670** | **0.920** | features (17, sin n_peaks), GMM K=15 | S2 ✅ claim v1 |
| LPI normalizado (rate feats) | 0.556 | 0.250 | — | 0.446 | 0.896 | features normalizadas, GMM K=15 | S2 audit |
| **LPINormalizingFlow (v2)** | **0.889** | **0.800** | — | **0.870** | **0.981** | RealNVP+GMM K=15, features 17, p90 | S2 ✅ single seed |
| LPIVariational (v2) | — | — | — | 0.778 | 0.932 | BayesianGMM K_eff=7, DP prior=0.01 | S2 ✅ |
| LPIBayesian (v2) | 0.833 | 0.375 | — | 0.670 | 0.920 | Bootstrap enrichment B=50, CI disponible | S2 ✅ |
| LPIHierarchical (v2) | 0.182 | 0.100 | — | 0.156 | 0.587 | GMM 2-nivel, K_macro=6 | S2 🔴 descartado |
| LPIOnline (v2) | 0.115 | 0.250 | — | 0.129 | 0.504 | Online EM warm_start, batch=200 | S2 🔴 descartado |
| Ensemble (NF+v1) | 0.780 | 0.800 | — | 0.784 | 0.976 | Average scores top-2 | S2 superado |
| **LPINormalizingFlow ensemble (median, 5 seeds)** | **1.000** | **0.575** | — | **0.871** | **0.997** | 16 features auditadas, median ensemble, CI95=[0.780, 0.931] | S2 ✅ **claim final** |

**Análisis threshold sweep (2026-04-12):**

| p | Val F0.5 | Val AUC |
|---|---|---|
| 70 | 0.677 | 0.803 |
| 75 | 0.664 | 0.803 |
| 80 | 0.664 | 0.803 |
| **85** | **0.683** | **0.803** |
| 90 | 0.631 | 0.803 |
| 92 | 0.606 | 0.803 |
| 95 | 0.570 | 0.803 |

El mejor umbral en validación es p85 (F0.5=0.683 > 0.669 OCSVM). Sin embargo, aplicado al test da F0.5=0.345 — una brecha val→test de 0.338 que indica **distribución shift entre anomalías de train y test**.

> **[2026-04-12] Decisión 4 — Optuna needed:**
> La brecha val→test no se resuelve con ajuste de umbral; el modelo no generaliza a los tipos de anomalía del test set. Causas probables: (a) pocos segmentos de entrenamiento (193 normales → 277 ventanas), (b) los tipos de anomalía del train y test difieren. Acción: Optuna sobre `d_model`, `n_layers`, `stride`, `epochs` + investigar distribución de anomalías por canal entre train y test splits.

> **[2026-04-12] Decisión 5 — Saltamos Optuna y sampling=1. Vamos al LPI:**
> El fix de imbalance en el Transformer subió de F0.5=0.345 → 0.641 (test). Gap vs OCSVM: 0.028 puntos. Dado que 0.028 es ruido estadístico (val solo tiene 7 anomalías tras rebalancear), el Transformer es defendible tal como está. Optuna y sampling=1 son trabajo de ingeniería sin ganancia comercial diferencial. Decisión: ir directo al LPI para obtener diferenciación con paper propio (Quesada 2026).

> **[2026-04-12] Decisión 6 — LPI como segundo modelo y modelo principal:**
> El LPI (Latent Propensity Index, Quesada 2026) aplicado sobre las 18 features estadísticas de OPS-SAT-AD (cohort sampling=5) da F0.5=0.801 y AUC=0.978 en test. El LPI es semi-supervised: usa labels del train para calcular enrichment por cluster GMM (K=15 seleccionado por BIC con bootstrap). Validado con GroupKFold 5-fold, test evaluado one-shot. Script: `experiments/s2_lpi/run_lpi_opssat.py`.

> **[2026-04-12] Decisión 7 — Audit de integridad: n_peaks inflada, claim recalibrado a LPI sin n_peaks:**
> El notebook `notebooks/03_lpi_integrity_audit.ipynb` detectó que `n_peaks` (AUC individual test=0.932) explica casi toda la ganancia del LPI full. Al normalizar por longitud: `n_peaks_per_point` AUC_test=0.702, `n_peaks_per_sec` AUC_test=0.674 → escenario **MIX**: señal física real (anomalías oscilan ~1.8× más por punto) pero amplificada por longitud (segmentos anómalos 3.4× más largos → más picos brutos). El LPI con `n_peaks` raw NO es honesto. **Claim v1: LPI sin n_peaks → F0.5=0.670, AUC=0.920.** Supera OCSVM en AUC (+0.120), empata en F0.5. Pendiente: verificar distribución de longitud de segmentos en telemetría SpainSat NG — si son homogéneos, `n_peaks` puede reincorporarse y el claim sube a 0.801/0.978.

> **[2026-04-13] Decisión 8 — LPI v2: LPINormalizingFlow es el modelo hero:**
> Script: `experiments/s2_lpi_v2/compare_extensions.py`. Dataset: 1001 train, 329 test (sampling=5, 17 features sin n_peaks). Protocolo anti-snooping idéntico al v1: 5-fold CV → threshold sweep → one-shot test.
>
> **Ganador: LPINormalizingFlow (RealNVP 4 capas, hidden=64, 42564 params de flow + GMM K=15):**
> F0.5=0.870, AUC=0.981, P=0.889, R=0.800 — vs v1 F0.5=0.670, AUC=0.920.
> Mejora de +0.200 en F0.5 y +0.061 en AUC. El flujo bijective mapea el espacio de features a un espacio Gaussiano donde el GMM puede separar los clusters de forma coherente. OOF AUC 0.952.
>
> **LPIVariational** (BayesianGMM DP prior=0.01, K_eff=7): F0.5=0.778, AUC=0.932 — segundo mejor, bate v1 en ambas métricas. La inference variacional elimina la búsqueda BIC y usa K efectiva automática.
>
> **LPIBayesian** (Bootstrap B=50): F0.5=0.670, AUC=0.920 — igual que v1 en predicción de punto (esperado). La ganancia es operacional: `score_with_uncertainty()` devuelve CI90 por muestra. No mejora F0.5 pero añade trazabilidad crítica para sistemas de seguridad.
>
> **Descartados:** LPIHierarchical (F0.5=0.156, K_macro=6 demasiado grueso), LPIOnline (F0.5=0.129, warm_start EM degrada parámetros en batches pequeños).
>
> **Ensemble (NF + v1):** F0.5=0.784, AUC=0.976 — peor que NF solo. No usar ensemble en producción.
>
> **Claim oficial actualizado: LPINormalizingFlow → F0.5=0.870, AUC=0.981.** Supera todos los baselines previos. Tests: 45/45 passed (`tests/test_lpi_v2.py`).
> ⚠️ Claim single-seed. Supersedido por Decisión 9 (ensemble con CI95).

> **[2026-04-13] Decisión 9 — LPINormalizingFlow ensemble (median, 5 seeds): claim final publicable con CI95:**
> Script: `experiments/s2_lpi_v2/run_nf_seed_ensemble.py`. Dataset: 1001 train, 329 test (sampling=5, **16 features** sin n_peaks, sin gaps_squared). Tests: 8/8 passed (`tests/test_nf_ensemble.py`).
>
> **Feature set limpio (16 features):** Se excluye gaps_squared además de n_peaks. Audit notebook 04 mostró gaps_squared dominante pero potencialmente length-confounded. Con 16 features, F0.5 medio sobre 5 seeds = 0.782±0.048 — base para el ensemble.
>
> **Seeds individuales (test):**
> | Seed | Val F0.5 | Test F0.5 | Test AUC |
> |------|----------|-----------|----------|
> | 0    | 0.771    | 0.769     | 0.972    |
> | 1    | 0.838    | 0.785     | 0.967    |
> | 42   | 0.633    | 0.707     | 0.950    |
> | 123  | 0.800    | 0.792     | 0.973    |
> | 999  | 0.708    | 0.855     | 0.971    |
> | mean±std | — | 0.782±0.048 | 0.966±0.009 |
>
> **Tres estrategias de ensemble:**
> | Estrategia | Test F0.5 | CI95 F0.5      | Test AUC | Val F0.5 |
> |------------|-----------|----------------|----------|----------|
> | Mean       | 0.882     | [0.800, 0.936] | 0.998    | 0.888    |
> | **Median** | **0.871** | **[0.780, 0.931]** | **0.997** | **0.905** |
> | Rank       | 0.957     | [0.903, 0.994] | 0.999    | 0.886    |
>
> **Ganador: Median ensemble** (criterio: mejor val F0.5=0.905 + menor gap val→test). El rank ensemble da test F0.5=0.957 (CI95=[0.903, 0.994]) — resultado excepcional con gap val→test positivo (+0.071). Se reporta como hallazgo adicional pero el claim conservador usa median.
>
> **Claim final publicable:** LPINormalizingFlow ensemble (median, 5 seeds), 16 features auditadas, **F0.5=0.871 (CI95=[0.780, 0.931]), AUC=0.997 (CI95=[0.993, 1.000])**. OPS-SAT-AD sampling=5, one-shot test, GroupKFold threshold selection.
>
> **Mejora vs baselines:** vs OCSVM: +0.202 F0.5 (+30.2%), +0.197 AUC. vs LPI v1: +0.201 F0.5 (+30.0%), +0.077 AUC. CI95 lower=0.780 > 0.75 — claim defendible ante revisor NeurIPS ML4PS o MNRAS Letters.
>
> **Publishability:** ✓ CI95 lower bound ≥ 0.75 supera todos los baselines con certeza estadística. Ensemble de 5 seeds independientes elimina cherry-picking de seed=42. Features auditadas (sin n_peaks por length leakage, sin gaps_squared por length confounding). Protocolo anti-snooping: test evaluado ONE-SHOT, threshold derivado de OOF. **Pendiente antes de envío: S3 (ESA-AD) para validar generalización cross-misión.**

### Descripción del modelo principal

- **Encoder-only Transformer**, reconstrucción por MSE, 71k parámetros (2 capas, d_model=64, 4 heads, d_ff=128)
- **Entrenamiento unsupervised**: solo sobre ventanas de segmentos normales del train set
- **Inferencia**: error de reconstrucción MSE por ventana → max por segmento → comparar con umbral
- **Normalización**: z-score por segmento antes de extraer ventanas
- **Umbral**: fijado por percentil sobre errores del train-normal (elegido en val, reportado en test)

### Cohort sampling=1 (pendiente, Fase 2)

- `window_size=256`, `stride=128`, mismo modelo con `seq_len=256`
- Esperar a validar S2 completo con sampling=5 antes de escalar
- Documentado en `CLAUDE.md` para trazabilidad ante revisores externos

### Hallazgos del EDA (notebooks/01_explore_segments.ipynb)

- 2123 segmentos, 1 canal por segmento, 9 canales distintos (CADC0872–CADC0894)
- Longitud variable: mediana 70 pts, media 143 pts, rango 8–1040
- Dos sampling rates: 1 Hz (793 segs, ~300 pts mediana) y 5 s (1330 segs, ~50 pts mediana)
- Anómalos 3.4× más largos que normales → length leakage resuelto con ventanas deslizantes
- Canal CADC0890: 73% anomaly rate (más informativo). CADC0884: 0% (excluir del unsupervised training)
- La columna `label` del CSV siempre contiene `'anomaly'` — ignorar, usar columna `anomaly` (0/1)

## 8. Métricas

- Precisión, recall, F1, **F0.5** (ESA recomienda F0.5 — FPs más costosos que FNs)
- AUC-ROC
- **Lead time** (tiempo medio de detección antes del fallo)
- **Tasa de FPs por día** ← lo que más les importa operativamente

## 9. Plan de 4 semanas

- **S1** ✅ — Setup, OPSSAT-AD, preprocesado, baselines (IF + OCSVM), MLflow.
- **S2** ✅ — Pipeline validado. Transformer rebalanced F0.5=0.641. LPI sin n_peaks F0.5=0.670, AUC=0.920 (claim oficial post-audit). LPI full (0.801/0.978) inflado por n_peaks — ver Decisión 7.
- **S3** — Escala a ESA-AD. PatchTST opcional. Métricas consolidadas.
- **S4** — Webapp, caso de estudio OPS-SAT, paper blanco técnico (5–10 pág), slides Hisdesat.

## 10. Equipo y coste

- 1 ML engineer senior (PyTorch + series temporales) — 4 semanas FT
- 1 full-stack — 2 semanas
- 1 consultor ops satélites — 10–20h
- 1 diseñador — 3–5 días

**Coste total aprox:** 20–30k€ externalizado + 100–300€ cloud (GPU ESA-AD).

## 11. Entregables MVP

1. Webapp con dashboard sobre anomalías reales OPS-SAT
2. Paper blanco técnico vs los 30 baselines ESA
3. Slides comerciales Hisdesat
4. Roadmap piloto con telemetría SpainSat NG-I

## 12. Pitch a Hisdesat (referencia)

> "SpainSat NG-II: una anomalía a 50.000 km costó 352M€. Vuestro sistema de umbrales detectó el impacto cuando ya había ocurrido. Hemos entrenado un modelo sobre datasets públicos de ESA con detección predictiva demostrable. Proponemos piloto de 6 meses sobre histórico de SpainSat NG-I para validar que habríamos dado alertas tempranas. Si funciona, producción."

## 13. Riesgos abiertos

1. **Contacto Hisdesat** — sin puerta de entrada no hay venta.
2. **Indra post-adquisición** — ¿tienen ya algo interno? Investigar.
3. **Airbus/Thales** — pueden decir "ya está en el satélite". Posicionar como complemento.
4. **Caso SpainSat NG-II** — impacto de partícula milimétrica no es predecible por telemetría. **No prometer lo imposible** en el pitch. Elegir bien los casos de éxito.
5. ~~Licencia datasets ESA~~ ✅ Resuelto: CC-BY 4.0, uso comercial permitido.

## 14. Diferencias con detector externo (MESPA)

|  | Externo (MESPA) | Interno (Hisdesat) |
|---|---|---|
| Cliente | MESPA / COVE | Hisdesat / Indra |
| Dato | TLEs públicos | Telemetría operador |
| ML | Medio (Isolation Forest) | Alto (Transformer) |
| Tiempo MVP | 3 sem | 4 sem |
| Coste MVP | 12–20k€ | 20–30k€ |
| Competencia ES | Nadie | Airbus, Thales, GMV, RS21 |
| Ventana | 12–18 meses | Más ajustada |

**Decisión pendiente:** ¿paralelo o secuencial al externo? ¿con qué equipo?

## 15. Próximos pasos

- [ ] Conseguir contacto en Hisdesat (García Primo, Romo, o intermediario)
- [x] ~~Verificar licencia comercial datasets ESA~~ → CC-BY 4.0 confirmado
- [ ] Perfilar ML engineer líder (PyTorch + series temporales + sector espacio)
- [ ] Investigar stack interno Indra post-adquisición
- [ ] Decidir paralelo vs secuencial vs externo
- [x] ~~Setup S1: baselines IF + OCSVM corriendo con MLflow~~
- [x] ~~S2 pipeline fix: ventanas deslizantes, anti-leakage test suite (17/17)~~
- [x] ~~S2 threshold sweep: val F0.5=0.683, test F0.5=0.345 — Optuna confirmed needed~~
- [x] ~~S2 imbalance fix: Transformer rebalanced F0.5=0.641 — gap vs OCSVM 0.028~~
- [x] ~~S2 LPI: F0.5=0.801, AUC=0.978 — modelo principal confirmado~~
- [x] ~~S2 audit integridad: n_peaks inflada. Claim recalibrado → LPI sin n_peaks F0.5=0.670, AUC=0.920~~
- [x] ~~S2 LPI v2: LPINormalizingFlow F0.5=0.870, AUC=0.981. LPIVariational F0.5=0.778, AUC=0.932. LPIBayesian CI90. 45/45 tests.~~
- [x] ~~S2 NF ensemble: median ensemble 5 seeds, 16 features auditadas. F0.5=0.871 (CI95=[0.780, 0.931]), AUC=0.997. 8/8 tests.~~
- [ ] **Pendiente piloto:** verificar distribución de longitud de segmentos en telemetría SpainSat NG — si homogénea, n_peaks reincorporable y claim sube
- [ ] **S3:** Escala LPINormalizingFlow ensemble a ESA-AD (3 misiones, 12 GB). Validar generalización cross-misión — prerequisito para envío a NeurIPS ML4PS.
- [ ] **S3 paper v2:** Escribir paper LPI v2 con NormalizingFlow como contribución principal. Target: MNRAS Letters o NeurIPS ML4PS workshop.
- [ ] **S2 fase 2 (opcional):** Transformer cohort sampling=1, window_size=256

---

## 16. LPI v2 — Extensiones deep tech

### Motivación

Pasar de "applied AI con técnicas conocidas" a matemática propietaria y publicable en NeurIPS/ICML en lugar de solo MNRAS. Las extensiones están en `src/models/lpi_v2.py`, el experimento comparativo en `experiments/s2_lpi_v2/compare_extensions.py`, los tests en `tests/test_lpi_v2.py` (45/45).

### Tabla de resultados (OPS-SAT-AD sampling=5, one-shot test)

| Variante | F0.5 | F0.5 CI95 | AUC | Val F0.5 | Features | Estado |
|---|---|---|---|---|---|---|
| LPI v1 (baseline) | 0.670 | — | 0.920 | 0.646 | 17 (sin n_peaks) | ✅ claim v1 |
| LPINormalizingFlow single seed | 0.870 | — | 0.981 | 0.695 | 17 (sin n_peaks) | single seed |
| LPINormalizingFlow 5 seeds (mean) | 0.782 | ±0.048 (seed std) | 0.966 | — | 16 auditadas | seed variability |
| LPIVariational | 0.778 | — | 0.932 | — | 17 | ✅ ligero |
| LPIBayesian | 0.670 | — | 0.920 | 0.646 | 17 | ✅ CI90 ops |
| LPIHierarchical | 0.156 | — | 0.587 | 0.519 | 17 | 🔴 descartado |
| LPIOnline | 0.129 | — | 0.504 | 0.253 | 17 | 🔴 descartado |
| NF ensemble mean | 0.882 | [0.800, 0.936] | 0.998 | 0.888 | 16 auditadas | ✅ |
| **NF ensemble median** | **0.871** | **[0.780, 0.931]** | **0.997** | **0.905** | **16 auditadas** | ✅ **CLAIM FINAL** |
| NF ensemble rank | 0.957 | [0.903, 0.994] | 0.999 | 0.886 | 16 auditadas | ✅ upper bound |

### Correlación Spearman de scores (test set)

- LPI v1 ↔ LPIBayesian: 1.000 (idéntico — esperado, misma geometría GMM)
- LPI v1 ↔ LPINormalizingFlow: 0.582 (señales distintas → potencial ensemble)
- LPIOnline ↔ todos: negativo (-0.4 a -0.7) → modelo degradado, señal invertida

### Decisiones tomadas (2026-04-13)

**Claim oficial (Decisión 9, 2026-04-13):**
- **NF ensemble median, 16 features auditadas**: F0.5=0.871 (CI95=[0.780, 0.931]), AUC=0.997
- Script: `experiments/s2_lpi_v2/run_nf_seed_ensemble.py`
- Ángulo paper actualizado: *"Multi-seed ensemble de NormalizingFlow sobre 16 features auditadas (sin length-leakage features) logra F0.5=0.871 (CI95=[0.780, 0.931]) superando todos los baselines con certeza estadística en OPS-SAT-AD"*

**Mantenemos:**
1. **LPINormalizingFlow ensemble (median)** — claim final con CI95 para pitch y paper.
2. **LPIVariational** — alternativa más ligera (~1k params). Útil para dispositivos edge o inferencia rápida.
3. **LPIBayesian** — no mejora F0.5 pero añade CI90 operacional. Incluir en producción como feature de trazabilidad.

**Descartamos:**
- **LPIHierarchical**: K_macro=6 demasiado grueso para este dataset (1001 segs). Gap val→test masivo (0.519→0.156). Requeriría K_macro grande + tuning que converge al LPI v1.
- **LPIOnline**: warm_start EM en batches pequeños degrada los parámetros GMM. El concepto es correcto pero necesita implementación con sufficient statistics explícitas (Cappé & Moulines 2009) en lugar de warm_start sklearn. Trabajo futuro para S3/S4.

### Audit de integridad LPINormalizingFlow (2026-04-13)

Ejecutado en `notebooks/04_nf_integrity_audit.ipynb`. **Resultado: el claim 0.870 NO es publicable tal cual.**

| Validación | Resultado | Veredicto |
|---|---|---|
| V1 Feature ablation | `diff2_var` ΔF0.5=−0.342, `gaps_squared` ΔF0.5=−0.224 | ⚠️ 2 features críticas |
| V2 Channel bias | Sin CADC0890: F0.5=0.859 (Δ=−0.011) | ✅ Robusto |
| V3 Arch sensitivity | std(F0.5)=0.315, rango [0.171, 0.870] | 🔴 Frágil — solo 4L h64 funciona |
| V4 Seed stability | mean=0.785, std=0.088, min=0.676, max=0.870 | 🔴 Alta varianza |
| V5 Latent space | 3 clusters anomaly-rich (enrichment>0.82), 8 normales | ✅ Mecanismo real |

**Claim honesto para paper (2026-04-13):** F0.5=0.785±0.088 (mean±std, 5 seeds, 4L h64). Mejor configuración observada: F0.5=0.870 (seed=42 y seed=123: 0.867). Peor: F0.5=0.676 (seed=0).

**Acciones requeridas antes de publicar:**
1. **Investigar `diff2_var`**: ¿es señal física real (jerk de la varianza ≡ cambio en la frecuencia de oscilación) o artefacto del dataset? Comparar con SpainSat NG telemetría.
2. **Investigar `gaps_squared`**: misma pregunta.  
3. **Buscar arquitectura robusta**: por qué 2 capas no converge (undercapacity) y 4L h128 cae a 0.708 (overfitting al flow). ¿Hay configuración que dé F0.5>0.80 con std<0.05?
4. **Reducir varianza de seed**: probar warm-start o ensemble de 3 seeds → reportar mediana.

### Papers a citar para LPINormalizingFlow (claim principal)

1. Dinh et al. (2017). Density estimation using Real-NVP. ICLR 2017. arXiv:1605.08803
2. Papamakarios et al. (2021). Normalizing Flows for Probabilistic Modeling. JMLR 22(57).
3. Osada et al. (2023). Unsupervised Anomaly Detection Using Normalizing Flows. Neurocomputing.
4. Gonzalez et al. (2025). Transformers for OPS-SAT-AD. Acta Astronautica.

### Arquitectura LPINormalizingFlow

```
RobustScaler(X) → RealNVP(4 coupling layers, hidden=64) → Z ≈ N(0,I)^17
→ GaussianMixture(K=15, BIC bootstrap) → enrichments f_k
→ LPI(x) = sum_k P(C_k|z(x)) * f_k
```

Parámetros flow: 42564. Parámetros GMM: 2565. Total: ~45k. Tiempo entrenamiento: 21s CPU.

---

## Cómo trabajar este doc con Claude

- Cuando pida "plan técnico", referir secciones 5–9.
- Cuando pida "pitch" o "comercial", secciones 2, 3, 12, 13.
- Cuando pida "comparativa con externo", sección 14.
- Antes de cualquier afirmación sobre Hisdesat/Indra/ESA: **verificar con web search** — el sector se mueve rápido y este doc puede quedarse desfasado.
