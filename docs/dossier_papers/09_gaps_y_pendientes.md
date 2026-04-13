# 09 — Gaps y Pendientes por Paper

> Lista priorizada de lo que falta para cada paper hasta submit.
> Prioridad: BLOQUEANTE > Alta > Media > Baja.

---

## Paper 1 — LPI original: CHIME/FRB + eROSITA (target: MNRAS Letters)

### Estado del draft

[VERIFICAR: Rodrigo confirma si existe un draft .tex o .md del Paper 1 en el repo o fuera de él, y en qué estado está (introducción, métodos, resultados, discusión).]

### Gaps identificados

#### BLOQUEANTES (sin esto no hay submit)

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P1-B1 | Datasets CHIME y eROSITA no están en el repo | Añadir scripts de descarga/preprocesado y verificar que los resultados son reproducibles | 1-2 días |
| P1-B2 | Número exacto de features usadas para cada dataset no documentado en el repo | Extraer del notebook o script de Paper 1 (si existe) | 0.5 días |
| P1-B3 | Referencias [VERIFICAR] en `08_papers_a_citar.md` (CHIME Catalog 2, eROSITA dwarfs) | Rodrigo confirma DOIs y títulos exactos | 0 días (info) |

#### Alta prioridad

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P1-A1 | Tabla de resultados cross-domain completa | Añadir IF, OCSVM, DAGMM para CHIME y eROSITA | 2-3 días |
| P1-A2 | Comparación con métodos del estado del arte en clasificación de FRBs | Buscar papers 2023-2025 que clasifiquen repeaters | 0.5 días |
| P1-A3 | Figura principal: scatter LPI score vs label para CHIME | [PENDIENTE: generar con matplotlib desde los OOF scores] | 0.5 días |
| P1-A4 | Ablation por número de componentes K (por qué K=15 y no K=5 o K=30) | Generar curva BIC media vs K para CHIME y eROSITA | 0.5 días |

#### Media prioridad

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P1-M1 | Análisis de robustez: ¿qué pasa si se usan diferentes ratios de clase rara? | Subsample repeaters, reevaluar | 1 día |
| P1-M2 | Discusión: limitaciones del LPI cuando ratio de clase rara es >50% (eROSITA) | Análisis teórico, sin código | 0.5 días |

#### Estimación total hasta submit

| Fase | Descripción | Esfuerzo estimado |
|---|---|---|
| Completar gaps BLOQUEANTES | P1-B1, B2, B3 | 2-3 días |
| Completar gaps de alta prioridad | P1-A1 a A4 | 4-5 días |
| Redacción y revisión del draft | Intro, Related Work, Methods, Results, Discussion | 5-7 días |
| Iteración con coautores | Revisión, ajustes | 2-3 días |
| Formato MNRAS (LaTeX) | Adaptar template | 1 día |
| **Total estimado** | | **~15-20 días efectivos** |

---

## Paper 3 — LPINormalizingFlow ensemble: OPS-SAT-AD (target: NeurIPS ML4PS workshop)

### Estado del claim

Claim final: **F0.5=0.871 (CI95=[0.780, 0.931]), AUC=0.997**. Publicable con CI95 verificado. Tests: 8/8 + 45/45 + 17/17.

### Gaps identificados

#### BLOQUEANTES (sin esto no hay submit en NeurIPS ML4PS)

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P3-B1 | **Validación en ESA-AD** — prerequisito explícito en CLAUDE.md para envío | Descargar ESA-AD (12 GB, 3 misiones KP Labs), ejecutar NF ensemble, reportar generalización cross-misión | 3-5 días (+ GPU time ~24-72h) |
| P3-B2 | **30 baselines KP Labs** no ejecutados sobre el mismo split | Clonar https://github.com/kplabs-pl/OPS-SAT-AD, ejecutar sobre sampling=5, añadir a tabla | 1-2 días |
| P3-B3 | diff2_var: ¿señal física o artefacto? | Consultar con experto de operaciones satelitales o buscar documentación técnica de OPS-SAT sobre CADC channels | 0.5 días (info) |

#### Alta prioridad

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P3-A1 | Gonzalez 2025 (Transformers, Acta Astronautica): extraer métricas exactas | Leer paper S0094576525006095 y añadir números a tabla comparativa | 0.5 días |
| P3-A2 | Ablation study completo con 16 features | Ejecutar ablation leave-one-out para las 14 features no críticas (V1 solo probó las 2 críticas) | 1-2 días |
| P3-A3 | Análisis de complejidad computacional | Medir tiempo de entrenamiento y inferencia por muestra para NF ensemble, LPI v1, OCSVM, IF; añadir tabla al paper | 0.5 días |
| P3-A4 | Figura latent space (t-SNE del espacio Z con coloreado por enrichment) | Extraer de notebook 04, pulir con matplotlib | 0.5 días |
| P3-A5 | Verificar correlación longitud ↔ gaps_squared | `pearson(dataset['gaps_squared'], segment_lengths)` — si < 0.3, gaps_squared se puede reincorporar | 0.5 días |

#### Media prioridad

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P3-M1 | DAGMM implementado y evaluado | Adaptar código público a feature dataset 16-dim | 1-2 días |
| P3-M2 | Deep SVDD implementado y evaluado | Adaptar a datos tabulares | 1-2 días |
| P3-M3 | Análisis cohort sampling=1 (Fase 2) | Ejecutar NF ensemble con window_size=256 sobre cohort 1 Hz | 3-4 días |
| P3-M4 | Buscar arquitectura NF más robusta (V3 falla con 2L y 4L h128) | Grid search n_layers ∈ {3,4,5}, hidden ∈ {32,64,96} con 5 seeds | 1-2 días |

#### Baja prioridad

| # | Gap | Acción requerida | Esfuerzo |
|---|---|---|---|
| P3-L1 | Anomaly Transformer (Xu 2022) | Requiere reescribir pipeline a series temporales (segments.csv) | 3-5 días |
| P3-L2 | n_peaks: verificar longitud segmentos SpainSat NG | Rodrigo consulta con Hisdesat/operador | 0 días (info) |

### Estimación total hasta submit NeurIPS ML4PS

**Deadline NeurIPS ML4PS:** [VERIFICAR: Rodrigo confirma el deadline del workshop — típicamente mayo-junio para NeurIPS diciembre]

| Fase | Descripción | Esfuerzo |
|---|---|---|
| ESA-AD validation (P3-B1) | Descargar, ejecutar, reportar | 3-5 días + GPU |
| 30 baselines KP Labs (P3-B2) | Clonar y ejecutar | 1-2 días |
| Ablation + timing (P3-A2, A3) | Código ligero | 1-2 días |
| Figuras (P3-A4) | matplotlib | 0.5 días |
| Redacción paper (4-6 páginas, NeurIPS ML4PS format) | Intro, Methods, Results, Conclusion | 4-5 días |
| Revisión interna | — | 1-2 días |
| Formato NeurIPS (LaTeX template) | — | 0.5 días |
| **Total estimado** | | **~12-18 días efectivos** |

---

## Resumen ejecutivo de pendientes

### PENDIENTE — requieren ejecución de código

| ID | Descripción |
|---|---|
| P1-B1 | Reproducibilidad CHIME/eROSITA: añadir scripts de descarga y preprocesado |
| P1-A1 | IF + OCSVM + DAGMM sobre CHIME y eROSITA |
| P1-A3 | Figura scatter LPI score vs label para CHIME |
| P1-A4 | Curva BIC vs K para datasets astrofísicos |
| P3-B1 | NF ensemble sobre ESA-AD (3 misiones, 12 GB) — requiere GPU |
| P3-B2 | 30 baselines KP Labs sobre OPS-SAT-AD sampling=5 |
| P3-A2 | Ablation leave-one-out 14 features restantes |
| P3-A3 | Timing: entrenamiento e inferencia por muestra de todos los modelos |
| P3-A5 | pearson(gaps_squared, segment_length) — 1 línea de código |
| 02-D | Contar N anómalos/normales exactos en cohort sampling=5 (train + test) |
| 06 | Extraer AUC individuales de features del notebook 03 (todos excepto n_peaks y gaps_squared) |

### VERIFICAR — requieren input de Rodrigo

| ID | Descripción |
|---|---|
| P1-B2 | Número y lista de features usadas en LPI v1 para CHIME y eROSITA |
| P1-B3 | DOI y título exacto del CHIME/FRB Catalog 2 |
| P1-B3 | DOI y título exacto del paper eROSITA dwarf galaxies |
| P1-B3 | Licencias de los catálogos CHIME y eROSITA |
| P3-B3 | Validación física de diff2_var con experto de operaciones OPS-SAT |
| P3-B3 | Estado del draft del Paper 1 (¿existe .tex fuera de este repo?) |
| P3-L2 | Longitud típica de segmentos en telemetría SpainSat NG-I (para decisión sobre n_peaks) |
| General | Deadline del workshop NeurIPS ML4PS (para priorizar Paper 1 vs Paper 3) |
| General | ¿Hay coautores? ¿Quien es "Quesada 2026"? |
