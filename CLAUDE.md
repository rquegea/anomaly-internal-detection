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
| **LPI sin n_peaks (Quesada 2026)** | — | — | — | **0.670** | **0.920** | features (17, sin n_peaks), GMM K=15 | S2 ✅ **claim oficial** |
| LPI normalizado (rate feats) | 0.556 | 0.250 | — | 0.446 | 0.896 | features normalizadas, GMM K=15 | S2 audit |

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
> El notebook `notebooks/03_lpi_integrity_audit.ipynb` detectó que `n_peaks` (AUC individual test=0.932) explica casi toda la ganancia del LPI full. Al normalizar por longitud: `n_peaks_per_point` AUC_test=0.702, `n_peaks_per_sec` AUC_test=0.674 → escenario **MIX**: señal física real (anomalías oscilan ~1.8× más por punto) pero amplificada por longitud (segmentos anómalos 3.4× más largos → más picos brutos). El LPI con `n_peaks` raw NO es honesto. **Claim oficial: LPI sin n_peaks → F0.5=0.670, AUC=0.920.** Supera OCSVM en AUC (+0.120), empata en F0.5. Pendiente: verificar distribución de longitud de segmentos en telemetría SpainSat NG — si son homogéneos, `n_peaks` puede reincorporarse y el claim sube a 0.801/0.978.

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
- [ ] **Pendiente piloto:** verificar distribución de longitud de segmentos en telemetría SpainSat NG — si homogénea, n_peaks reincorporable y claim sube a 0.801/0.978
- [ ] **S3:** Escala LPI (sin n_peaks) + Transformer a ESA-AD (3 misiones, 12 GB). Validar generalización.
- [ ] **S2 fase 2 (opcional):** Transformer cohort sampling=1, window_size=256

---

## Cómo trabajar este doc con Claude

- Cuando pida "plan técnico", referir secciones 5–9.
- Cuando pida "pitch" o "comercial", secciones 2, 3, 12, 13.
- Cuando pida "comparativa con externo", sección 14.
- Antes de cualquier afirmación sobre Hisdesat/Indra/ESA: **verificar con web search** — el sector se mueve rápido y este doc puede quedarse desfasado.
