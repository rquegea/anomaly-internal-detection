# 05 — Datasets

> Descripción completa de los 3 datasets usados en este proyecto.
> Paper 1 (LPI original): CHIME/FRB + eROSITA. Paper 3 (NF ensemble): OPS-SAT-AD.

---

## 1. CHIME/FRB Catalog 2 (Paper 1)

| Campo | Valor |
|---|---|
| Nombre completo | CHIME/FRB Catalog 2 |
| Dominio | Radioastronomía — Fast Radio Bursts (FRBs) |
| Fuente | CHIME/FRB Collaboration |
| URL | [VERIFICAR: Rodrigo confirma DOI/URL exacta del catálogo] |
| Licencia | [VERIFICAR: Rodrigo confirma licencia — posiblemente CC-BY o acceso abierto] |
| N total | 3 641 eventos FRB |
| N clase rara | 83 FRBs repetidores (repeaters) |
| Ratio clase rara | 2.3% |
| Dimensión | [PENDIENTE: número de features usadas en LPI v1 para CHIME] |
| Split | 5-fold cross-validation (no train/test fijo) |
| Resultado LPI v1 | AUC = 0.760 ± 0.048 (5-fold CV) |

### Features utilizadas

[PENDIENTE: listar las features del catálogo CHIME usadas en LPI v1 — espectro, DM, flujo, etc.]

### Descripción del problema

Los FRBs son pulsos de radio de origen extragaláctico de milisegundos de duración. La mayoría son eventos únicos (non-repeaters). Una minoría (~2%) son repeaters: la misma fuente emite múltiples bursts. Identificar repeaters es científicamente prioritario (implican diferente mecanismo físico) pero difícil porque los catálogos son altamente desbalanceados.

LPI detecta repeaters como clase rara con AUC=0.760, superando IsolationForest y OneClassSVM sin labels de clase (excepto para el enrichment).

---

## 2. eROSITA Dwarf Galaxies (Paper 1)

| Campo | Valor |
|---|---|
| Nombre completo | eROSITA dwarf galaxies catalog |
| Dominio | Astronomía de rayos X — galaxias enanas con posibles agujeros negros de masa intermedia (IMBHs) |
| Fuente | eROSITA all-sky survey |
| URL | [VERIFICAR: Rodrigo confirma DOI/URL] |
| Licencia | [VERIFICAR: Rodrigo confirma licencia] |
| N total | 169 galaxias enanas |
| N clase rara | 95 candidatas IMBH |
| Ratio clase rara | 56.2% |
| Dimensión | [PENDIENTE: número de features usadas en LPI v1 para eROSITA] |
| Split | 5-fold cross-validation |
| Resultado LPI v1 | AUC = 0.708 ± 0.103 (5-fold CV) |

### Descripción del problema

Las IMBHs (Intermediate Mass Black Holes, $10^3 - 10^5 M_\odot$) son difíciles de distinguir de otras fuentes de rayos X en galaxias enanas. El dataset tiene un ratio inusual de clase rara (56%) porque los autores pre-seleccionaron candidatas. LPI en este contexto actúa como clasificador de confirmación, no de detección desde fondo.

---

## 3. OPS-SAT-AD (Paper 3)

### Descripción general

| Campo | Valor |
|---|---|
| Nombre completo | OPS-SAT Anomaly Detection Dataset |
| Dominio | Telemetría de satélite — nanosatélite CubeSat de ESA |
| Fuente | KP Labs + ESA OPS-SAT mission |
| URL Zenodo (versión original) | https://doi.org/10.5281/zenodo.12588359 |
| URL Zenodo (versión actualizada, abr 2025) | https://zenodo.org/records/15108715 |
| URL código oficial (30 baselines) | https://github.com/kplabs-pl/OPS-SAT-AD |
| Paper Nature Scientific Data | DOI: 10.1038/s41597-025-05035-3 |
| Preprint arXiv | https://arxiv.org/abs/2407.04730 |
| Licencia | CC-BY 4.0 — uso comercial permitido citando autores |
| Cita requerida | Ruszczak et al. (2024). Scientific Data, Springer Nature. |

### Estructura del dataset

El dataset contiene dos archivos:

**dataset.csv** — features pre-computadas por segmento

| Columna | Tipo | Descripción |
|---|---|---|
| anomaly | int | Etiqueta de clase: 0=normal, 1=anómalo |
| train | int | Split: 1=train, 0=test |
| channel | str | Canal de telemetría (CADC0872–CADC0894) |
| sampling | int | Frecuencia de muestreo: 1 (1 Hz) o 5 (1 muestra cada 5 s) |
| mean | float | Media de la serie temporal del segmento |
| std | float | Desviación estándar |
| min | float | Mínimo |
| max | float | Máximo |
| range | float | max - min |
| median | float | Mediana |
| q25, q75 | float | Cuartiles 25 y 75 |
| iqr | float | Rango intercuartil |
| skewness | float | Asimetría |
| kurtosis | float | Curtosis |
| n_peaks | float | Número de picos (descartada por leakage — ver `06_audit_trail.md`) |
| entropy | float | Entropía de Shannon |
| autocorr | float | Autocorrelación lag-1 |
| diff_mean | float | Media de primeras diferencias |
| diff_var | float | Varianza de primeras diferencias |
| diff2_mean | float | Media de segundas diferencias |
| diff2_var | float | Varianza de segundas diferencias (feature más crítica en NF) |
| gaps_mean | float | Media de gaps temporales |
| gaps_squared | float | Cuadrado de suma de gaps (descartada por leakage — ver audit) |

Total features originales: 18. Features usadas en LPI v1 (sin n_peaks): 17. Features usadas en NF ensemble (sin n_peaks, sin gaps_squared): 16.

**segments.csv** — series temporales crudas

| Columna | Tipo | Descripción |
|---|---|---|
| segment_id | int | Identificador del segmento |
| timestamp | float | Marca de tiempo relativa (segundos) |
| value | float | Valor de telemetría |
| anomaly | int | Etiqueta del segmento (0/1) |
| train | int | Split |
| channel | str | Canal |
| sampling | int | Frecuencia |

### Cohorts por sampling rate

| Cohort | Sampling | N total segs | N train | N test | Mediana longitud | Uso en este proyecto |
|---|---|---|---|---|---|---|
| sampling=5 | 1 muestra / 5 s | 1 330 | 1 001 | 329 | ~50 pts | S2, S3 (principal) |
| sampling=1 | 1 Hz | 793 | [PENDIENTE] | [PENDIENTE] | ~300 pts | S2 Fase 2 (pendiente) |

### Canales de telemetría

| Canal | Anomaly rate | Notas |
|---|---|---|
| CADC0890 | 73% | Más informativo — alto rate de anomalías |
| CADC0884 | 0% | Solo normal — excluir del entrenamiento unsupervised |
| CADC0872–CADC0894 | Variable | 9 canales distintos en total |

### Distribución de clases (cohort sampling=5)

| Clase | N train | N test |
|---|---|---|
| Normal | [PENDIENTE: contar exacto del CSV] | [PENDIENTE] |
| Anómalo | [PENDIENTE] | [PENDIENTE] |
| Ratio anomalía | [PENDIENTE] | [PENDIENTE] |

### Hallazgos EDA (notebook 01)

- 2123 segmentos totales (ambos cohorts)
- Longitud variable: mediana=70 pts, media=143 pts, rango=8–1040 pts
- Segmentos anómalos son 3.4× más largos que normales (mediana 184 vs 54 pts, cohort sampling=5) → **length leakage** resuelto con ventanas deslizantes
- Dos sampling rates distintas: 1 Hz (793 segs) y 5 s (1330 segs)
- La columna `label` del CSV siempre es `'anomaly'` — ignorar; usar columna `anomaly` (0/1)

---

## 4. Tabla comparativa de los 3 datasets

| Propiedad | CHIME/FRB | eROSITA dwarfs | OPS-SAT-AD (sampling=5) |
|---|---|---|---|
| Dominio | Radioastronomía | Astrofísica X | Telemetría satelital |
| N total | 3 641 | 169 | 1 330 |
| N clase rara | 83 (2.3%) | 95 (56.2%) | [PENDIENTE] |
| Ratio clase rara | 2.3% | 56.2% | [PENDIENTE]% |
| Dimensión original | [PENDIENTE] | [PENDIENTE] | 18 features |
| Dimensión usada | [PENDIENTE] | [PENDIENTE] | 16 features |
| Split | 5-fold CV | 5-fold CV | train/test fijo |
| Licencia | [VERIFICAR] | [VERIFICAR] | CC-BY 4.0 |
| Paper | Paper 1 | Paper 1 | Paper 3 |
| Modelo principal | LPI v1 | LPI v1 | NF ensemble median |
| AUC | 0.760 ± 0.048 | 0.708 ± 0.103 | 0.997 [0.993, 1.000] |

---

## 5. Datos no disponibles en este repo

Los datasets CHIME/FRB y eROSITA **no están incluidos** en el repositorio. Solo está OPS-SAT-AD en `reference/data/`. Para reproducir los resultados del Paper 1 es necesario descargar los catálogos de sus fuentes originales.

[VERIFICAR: Rodrigo confirma URIs oficiales de CHIME Catalog 2 y eROSITA dwarfs, y si el preprocesado está documentado en otro notebook/script no incluido en este repo.]

---

## 6. Cómo citar OPS-SAT-AD

```
Ruszczak, B., Kotowski, K., Evans, D., & Nalepa, J. (2025).
The OPS-SAT benchmark for detecting anomalies in satellite telemetry.
Scientific Data, Springer Nature.
DOI: 10.1038/s41597-025-05035-3
```

Dataset en Zenodo (versión original):
```
OPSSAT-AD - anomaly detection dataset for satellite telemetry.
Zenodo. DOI: 10.5281/zenodo.12588359
```
