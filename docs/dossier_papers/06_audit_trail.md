# 06 — Audit Trail: Hallazgos de Integridad

> Reproduce los hallazgos clave de los notebooks 03 y 04.
> Esta sección es lo que un revisor de NeurIPS quiere ver para creerse el rigor.

---

## Notebook 03 — Audit de integridad LPI v1: descubrimiento de n_peaks

**Archivo:** `notebooks/03_lpi_integrity_audit.ipynb`
**Fecha:** 2026-04-12
**Pregunta:** ¿Es el AUC=0.978 del LPI full honesto o está inflado por alguna feature?

### Hallazgo 1: n_peaks domina el LPI full

Análisis de AUC individual por feature (área bajo la curva ROC usando solo esa feature como score):

| Feature | AUC (test) |
|---|---|
| n_peaks | 0.932 |
| diff2_var | [PENDIENTE: extraer del notebook 03] |
| entropy | [PENDIENTE] |
| iqr | [PENDIENTE] |
| std | [PENDIENTE] |
| gaps_squared | [PENDIENTE: verificar si aparece en nb 03 o solo en nb 04] |

`n_peaks` tiene AUC individual de 0.932 — prácticamente igual al AUC total del LPI sin n_peaks (0.920). Esto sugiere que el LPI full (AUC=0.978) está siendo arrastrado principalmente por n_peaks.

### Hallazgo 2: n_peaks es length-confounded (escenario MIX)

| Métrica | Valor |
|---|---|
| Ratio de longitud anómalo/normal (mediana) | 3.4× (184 pts vs 54 pts, sampling=5) |
| n_peaks raw: AUC test | 0.932 |
| n_peaks_per_point (n_peaks / longitud segmento): AUC test | 0.702 |
| n_peaks_per_sec (n_peaks / duración en segundos): AUC test | 0.674 |

**Interpretación:** La caída de AUC de 0.932 → 0.702 al normalizar por longitud confirma que n_peaks está amplificado por la mayor duración de los segmentos anómalos. Sin embargo, el AUC no cae a 0.5 (azar), lo que indica que existe señal física real: los segmentos anómalos oscilan genuinamente más por punto (~1.8× según scatter plot del notebook 03 — ver `plots/03_decisive_scatter.png`).

**Veredicto — Escenario MIX:**
- Señal real: anomalías oscilan más por punto (física del satélite)
- Amplificador artefactual: segmentos anómalos son 3.4× más largos → más picos brutos

**Decisión (Decisión 7 de CLAUDE.md):** n_peaks excluida del claim publicable. LPI sin n_peaks → F0.5=0.670, AUC=0.920.

### Evidencia visual

- `plots/03_q1_channel_bias.png` — distribución de anomalías por canal
- `plots/03_q2_feature_auc.png` — AUC individual por feature
- `plots/03_q3_ablation.png` — ablation del LPI con/sin n_peaks
- `plots/03_decisive_scatter.png` — scatter n_peaks vs longitud, coloreado por label

### Implicación para producción

Si en la telemetría de SpainSat NG los segmentos tienen longitud homogénea (constraintoperacional del sistema de TM/TC), n_peaks puede reincorporarse y el claim sube a F0.5=0.801, AUC=0.978. [VERIFICAR: Rodrigo consulta con el operador.]

---

## Notebook 04 — Audit de integridad LPINormalizingFlow

**Archivo:** `notebooks/04_nf_integrity_audit.ipynb`
**Fecha:** 2026-04-13
**Pregunta:** ¿Es el claim F0.5=0.870 del NF single seed publicable?

**Resultado general: NO como single seed.** Alta varianza de seed (V4). Resuelto con ensemble de 5 seeds (Decisión 9).

### Validación V1 — Feature ablation

| Feature eliminada | F0.5 sin esa feature | Delta F0.5 |
|---|---|---|
| diff2_var | 0.528 | **-0.342** |
| gaps_squared | 0.646 | **-0.224** |
| diff2_mean | ~0.83 | < -0.04 |
| Resto de features | > 0.83 | < -0.04 |

**Hallazgo:** `diff2_var` (varianza de segundas diferencias = "jerk" de la varianza) es la feature más crítica. Al eliminarla, F0.5 colapsa 0.342 puntos. Esto plantea la pregunta: ¿es señal física real o artefacto?

**Pregunta abierta sobre diff2_var:** Las segundas diferencias capturan cambios en la tasa de cambio de la señal (jerk). En telemetría de satélites, un cambio en la frecuencia de oscilación puede indicar un fallo emergente (ej. degradación de un giroscopio). Sin acceso a la documentación física de OPS-SAT, no es posible confirmar si diff2_var refleja un fenómeno real o un artefacto de la segmentación variable. [PENDIENTE: investigar con experto de operaciones satelitales]

**Hallazgo sobre gaps_squared:** ΔF0.5=-0.224 al eliminar gaps_squared. Como el feature mide la suma de gaps temporales al cuadrado, es potencialmente correlacionado con la longitud del segmento → posible length confounding indirecto. Excluida de las 16 features del claim final.

**Veredicto V1:** ⚠️ 2 features críticas. El claim de NF depende fuertemente de diff2_var (que necesita validación física) y gaps_squared (excluida por leakage potencial). El claim debe reportarse con las 16 features auditadas.

### Validación V2 — Channel bias

Experimento: reentrenar y evaluar el NF sin incluir datos del canal CADC0890 (73% anomaly rate, el más informativo).

| Métrica | Con CADC0890 | Sin CADC0890 | Delta |
|---|---|---|---|
| F0.5 (test) | 0.870 | 0.859 | -0.011 |

**Veredicto V2:** ✅ El modelo es robusto a la exclusión del canal más informativo. La señal no depende de un único canal. Delta = -0.011, despreciable.

### Validación V3 — Sensibilidad de arquitectura

Se probaron 4 configuraciones de RealNVP (variando n_layers y flow_hidden):

| Arquitectura | F0.5 |
|---|---|
| 2L h32 | [PENDIENTE: extraer del notebook 04] |
| **4L h64 (elegida)** | **0.870** |
| 4L h128 | 0.708 (overfitting del flow) |
| 2L h64 | [PENDIENTE: valor del notebook 04] |

| Estadístico | Valor |
|---|---|
| std(F0.5) sobre configuraciones | 0.315 |
| Rango | [0.171, 0.870] |

**Veredicto V3:** 🔴 El modelo es frágil. Solo la configuración 4L h64 produce resultados aceptables. 2 capas: undercapacity (no puede modelar la distribución). 4L h128: overfitting del flow (memoriza el train, pierde generalización). Esto limita las afirmaciones sobre la generalidad de la arquitectura.

**Implicación para el paper:** Reportar explícitamente que se realizó búsqueda de arquitectura y que 4L h64 fue la seleccionada. Esto es común en papers de NF pero debe documentarse para evitar acusaciones de cherry-picking.

### Validación V4 — Estabilidad de seed

Se entrenó el NF con 5 seeds distintas (misma configuración 4L h64):

| Seed | F0.5 (test) |
|---|---|
| 0 | 0.769 |
| 1 | 0.785 |
| 42 | 0.707 |
| 123 | 0.792 |
| 999 | 0.855 |
| mean | 0.782 |
| std | **0.048** |
| min | 0.707 |
| max | 0.855 (≠ 0.870 del single seed: diferente feature set — 17 vs 16) |

**Veredicto V4:** 🔴 Alta varianza de seed. std=0.048 significa que el intervalo de confianza del single seed abarca ~±2std = [0.694, 0.878]. El valor 0.870 es el extremo superior del rango observable. No publicable como resultado de punto.

**Resolución:** Ensemble de 5 seeds con mediana → F0.5=0.871, CI95=[0.780, 0.931]. La mediana es resistente a seeds outliers. El CI95 bootstrap confirma que el resultado no depende de seed lucky. (Decisión 9, `experiments/s2_lpi_v2/run_nf_seed_ensemble.py`)

### Validación V5 — Espacio latente (interpretabilidad)

Análisis del espacio latente Z aprendido por el NF (seed=42, configuración 4L h64):

| Cluster GMM | Enrichment f_k | Interpretación |
|---|---|---|
| Clusters 1–3 | > 0.82 | Anomaly-rich (3 clusters) |
| Clusters 4–15 | ~0.0 | Normal (12 clusters) |

**Veredicto V5:** ✅ El mecanismo es real. El flow aprende a separar anomalías en 3 clusters compactos en el espacio latente, mientras que los datos normales se distribuyen en los 12 clusters restantes. Esto es coherente con la hipótesis de que el NF mejora la separabilidad geométrica.

**Visualización:** `notebooks/04_nf_integrity_audit.ipynb` contiene t-SNE del espacio latente coloreado por enrichment y por label real.

---

## Hallazgo V6 — gaps_squared: leakage confirmada (2026-04-13)

Este hallazgo emergió durante la preparación del ensemble de 5 seeds, después de los audits V1–V5.

**Mecanismo:**
```
longitud del segmento
    → más timestamps → mayor suma de gaps
    → gaps_squared ∝ (suma gaps)^2 ∝ longitud^2
```

gaps_squared codifica información de longitud de forma amplificada (por el cuadrado). Dado que la longitud correlaciona con la clase (anómalos 3.4× más largos), gaps_squared puede actuar como proxy de longitud.

**Evidencia — correlación calculada (sampling=5, N=1330):**

| Métrica | Valor | Interpretación |
|---|---|---|
| Pearson(gaps_squared, len) — todos los segmentos | 0.506 | Correlación moderada |
| **Spearman(gaps_squared, len) — todos los segmentos** | **0.966** | Correlación de rango casi perfecta — **supera umbral 0.7** |
| Pearson(gaps_squared, len) — solo normales (N=1169) | **0.9998** | Función determinista de la longitud: sin señal propia |
| Pearson(gaps_squared, len) — solo anómalos (N=161) | 0.235 | Señal real presente (anomalías rompen la relación len→gaps) |

**Interpretación:**
- En normales, gaps_squared ≈ f(len) exacta. Aporta cero información discriminativa propia.
- En anómalos, la correlación cae a 0.235: las anomalías producen gaps temporales irregulares que se alejan del comportamiento normal. Hay señal física real.
- Pero esa señal real no es separable del confounding de longitud sin normalización explícita (`gaps_squared / len^2`).

**Veredicto:** 🔴 **Leakage confirmada.** Spearman=0.966 supera el umbral de 0.7 definido en el protocolo de audit. La señal existe (anomalías genuinamente producen gaps irregulares) pero está mezclada con longitud de manera no separable con la feature tal como está definida.

**Decisión (Decisión 9 — confirmada):** gaps_squared excluida de las 16 features del claim final. Exclusión correcta y publicable. Para reincorporarla en trabajos futuros: usar `gaps_squared / len^2` normalizado y re-auditar.

---

## Resumen de veredictos

| Validación | Resultado | Impacto en el paper |
|---|---|---|
| V1 — Feature ablation | ⚠️ diff2_var crítica, gaps_squared excluida | Reportar ablation en paper; pedir validación física de diff2_var |
| V2 — Channel bias | ✅ Robusto | Mencionar como evidencia de generalidad |
| V3 — Arch sensitivity | 🔴 Solo 4L h64 funciona | Documentar búsqueda de arquitectura en Appendix |
| V4 — Seed stability | 🔴 std=0.048 | Resuelto con ensemble median 5 seeds |
| V5 — Latent space | ✅ 3 anomaly clusters | Incluir figura t-SNE en el paper (interpretabilidad) |
| V6 — gaps_squared | 🔴 Leakage confirmada (Spearman=0.966 con len; Pearson normal=0.9998) | Excluida. Señal real pero no separable del confounding sin normalizar. |

---

## Tests de anti-leakage formales

Estos tests corren en CI y verifican las garantías de integridad a nivel de código:

| Test | Fichero | Verifica |
|---|---|---|
| `test_window_size_is_constant` | `test_no_length_leakage.py:45` | Todas las ventanas tienen shape (window_size,) |
| `test_normal_anomaly_lengths_identical` | `test_no_length_leakage.py:67` | Distribución de longitudes idéntica para clases 0 y 1 |
| `test_no_cv_leakage` | `test_lpi_v2.py:210` | Índices train/val estrictamente disjuntos en CV |
| `test_sampling_rate_disjoint` | `test_no_length_leakage.py:142` | Cohorts sampling=1 y =5 sin intersección |
