# 03 — Metodología: LPI v1 (Latent Propensity Index)

> Formulación matemática del algoritmo Quesada 2026 implementado en `src/models/lpi.py`.

---

## 1. Formulación matemática

### 1.1 Notación

Sea $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ el conjunto de entrenamiento, donde:
- $\mathbf{x}_i \in \mathbb{R}^d$ es el vector de features del objeto $i$
- $y_i \in \{0, 1\}$ es la etiqueta, con $y_i = 1$ marcando la clase rara (anomalía)
- $N^+ = |\{i : y_i = 1\}| \ll N$ (clase rara)

### 1.2 Paso 1 — Estandarización

Se aplica RobustScaler (mediana y rango intercuartil):

$$\tilde{x}_{ij} = \frac{x_{ij} - \text{median}_j}{\text{IQR}_j}$$

donde median_j e IQR_j se estiman sobre los datos de entrenamiento. Se usa RobustScaler en lugar de StandardScaler porque la telemetría de satélites contiene outliers transitorios que inflarían la varianza con StandardScaler.

### 1.3 Paso 2 — Selección de K por BIC bootstrap

Para eliminar la influencia de la inicialización aleatoria del GMM en la selección de K, se promedia el BIC sobre $B$ remuestras bootstrap:

$$K^* = \arg\min_{K \in \mathcal{K}} \frac{1}{B} \sum_{b=1}^B \text{BIC}(\hat{\theta}_K^{(b)}, \mathcal{D}^{(b)})$$

donde:
- $\mathcal{K} = \{2, 3, \ldots, 15\}$ (rango de búsqueda)
- $\mathcal{D}^{(b)}$ = resample bootstrap del 80% del train con reemplazo
- $\hat{\theta}_K^{(b)}$ = parámetros GMM ajustados sobre $\mathcal{D}^{(b)}$ con $K$ componentes
- $\text{BIC}(\hat{\theta}, \mathcal{D}) = \log(N) \cdot p - 2 \mathcal{L}(\hat{\theta}; \mathcal{D})$
- $p$ = número de parámetros del GMM con covarianza full: $K(1 + d + d(d+1)/2) - 1$

En la implementación (`src/models/lpi.py:78`): $B = 20$, cada resample usa el 80% del train.

### 1.4 Paso 3 — Ajuste final del GMM

Se ajusta un GMM con $K^*$ componentes sobre todos los datos de entrenamiento:

$$p(\mathbf{x}) = \sum_{k=1}^{K^*} \pi_k \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

con covarianza full, n_init=5 (para evitar óptimos locales), max_iter=300 y reg_covar=$10^{-6}$.

### 1.5 Paso 4 — Enrichment por cluster

El enrichment $f_k$ de cada cluster mide la concentración de clase rara en ese cluster:

$$f_k = \frac{\sum_{i=1}^N \mathbf{1}[\hat{c}_i = k] \cdot y_i}{\sum_{i=1}^N \mathbf{1}[\hat{c}_i = k]}$$

donde $\hat{c}_i = \arg\max_k P(C_k \mid \mathbf{x}_i)$ es la asignación hard al cluster más probable.

Si el cluster $k$ está vacío, $f_k = 0$.

**Propiedad clave:** $f_k \in [0, 1]$ y $f_k = 0$ para clusters sin ningún objeto de clase rara.

### 1.6 Paso 5 — LPI score (inferencia)

El score LPI de un objeto nuevo $\mathbf{x}$ es una combinación laxa de responsabilidades GMM ponderadas por enrichment:

$$\text{LPI}(\mathbf{x}) = \sum_{k=1}^{K^*} P(C_k \mid \mathbf{x}) \cdot f_k$$

donde $P(C_k \mid \mathbf{x}) = \frac{\pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j} \pi_j \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$ son las responsabilidades soft del GMM (Bayes posterior).

**Propiedad:** $\text{LPI}(\mathbf{x}) \in [0, 1]$. Un objeto que cae enteramente en clusters ricos en anomalías obtiene LPI $\approx 1$.

### 1.7 Paso 6 — Decisión binaria

$$\hat{y} = \mathbf{1}[\text{LPI}(\mathbf{x}) \geq \tau]$$

donde $\tau$ se selecciona por sweep sobre percentiles $\{70, 75, 80, 85, 90, 92, 95\}$ del score sobre el validation set (OOF scores del CV de 5-fold). Métrica objetivo: F0.5.

---

## 2. Pseudocódigo del algoritmo completo

```
Algorithm LPI_fit(X_train, y_train, K_range=[2..15], B=20, n_folds=5):

  # 1. Scale
  scaler = RobustScaler().fit(X_train)
  X_sc = scaler.transform(X_train)

  # 2. BIC bootstrap K selection
  bic_acc[k] = [] for k in K_range
  for b in 1..B:
    idx = random_choice(len(X_sc), size=0.8*len(X_sc), replace=True)
    X_boot = X_sc[idx]
    for k in K_range:
      gmm_b = GaussianMixture(k, covariance='full').fit(X_boot)
      bic_acc[k].append(gmm_b.bic(X_boot))
  K_star = argmin_k mean(bic_acc[k])

  # 3. Final GMM
  gmm = GaussianMixture(K_star, n_init=5, max_iter=300, reg_covar=1e-6)
  gmm.fit(X_sc)

  # 4. Enrichment
  assignments = gmm.predict(X_sc)         # hard assignments
  for k in 0..K_star-1:
    mask = (assignments == k)
    f[k] = mean(y_train[mask]) if any(mask) else 0.0

  return (scaler, gmm, f)


Algorithm LPI_score(x, scaler, gmm, f):
  x_sc = scaler.transform(x)
  R = gmm.predict_proba(x_sc)             # shape (n_samples, K_star)
  return R @ f                            # shape (n_samples,)


Algorithm LPI_threshold_selection(X_train, y_train, X_val, y_val,
                                  percentiles=[70,75,80,85,90,92,95]):
  # OOF CV for threshold
  oof_scores = fit_predict_cv(X_train, y_train, cv=5)
  best_tau, best_f05 = None, -inf
  for p in percentiles:
    tau = percentile(oof_scores, p)
    preds = (oof_scores >= tau).astype(int)
    f05 = fbeta_score(y_train, preds, beta=0.5)
    if f05 > best_f05:
      best_f05, best_tau = f05, tau

  # One-shot test evaluation
  model = LPI_fit(X_train, y_train)
  test_scores = LPI_score(X_test, ...)
  preds_test = (test_scores >= best_tau).astype(int)
  report_metrics(y_test, preds_test, test_scores)
```

---

## 3. Justificación de decisiones de diseño

### RobustScaler vs StandardScaler

La telemetría de satélites contiene transitorios (spikes) causados por eventos electromagnéticos o comandos. StandardScaler inflaría la desviación estándar, desensibilizando el GMM. RobustScaler usa mediana e IQR, que son resistentes a outliers extremos.

### BIC vs AIC vs número fijo de componentes

El BIC penaliza modelos complejos más fuertemente que el AIC ($\log N$ vs $2$), lo que tiende a seleccionar modelos más parsimoniosos. En datasets de tamaño moderado (N~1000) esto previene overfitting del GMM. El bootstrap promedia sobre la varianza de inicialización aleatoria del GMM, produciendo una estimación más estable de K*.

### K range (2, 15)

El límite inferior K=2 permite separar al menos dos poblaciones (normal vs anómala). K=15 es el máximo que asegura convergencia robusta con N=1001 muestras de train (>65 muestras por componente en media). En la práctica, BIC selecciona K=15 para OPS-SAT-AD sampling=5.

### Enrichment hard vs soft

Se usa asignación hard ($\hat{c}_i = \arg\max P(C_k|\mathbf{x}_i)$) para el enrichment y asignación soft ($P(C_k|\mathbf{x})$) para el score. Este diseño asimétrico es intencional: el enrichment debe ser interpretable (fracción de anomalías en el cluster), mientras que el score debe ser continuo y diferenciable para maximizar AUC.

### F0.5 como métrica objetivo

ESA recomienda F0.5 (peso mayor a precision) porque los falsos positivos tienen mayor coste operacional que los falsos negativos: una alarma falsa interrumpe operaciones satelitales; una anomalía perdida puede recuperarse si es detectada en el siguiente ciclo.

---

## 4. Diagrama de flujo del pipeline

```
dataset.csv (features por segmento)
        |
        v
[Filtro sampling=5] --> 1330 segmentos
        |
        v
[Split train/test] -- GroupKFold por segmento
        |
     train (1001)          test (329)
        |                      |
        v                      |
[RobustScaler.fit_transform]   |
        |                      |
        v                      |
[BIC bootstrap K*=15]          |
        |                      |
        v                      |
[GMM.fit(X_train_sc)]          |
        |                      |
        v                      |
[Enrichments f_k]              |
        |                      |
        v                      |
[5-fold OOF scores]            |
        |                      |
        v                      |
[Threshold sweep: argmax F0.5]  |
        |                      |
        +-------> ONE-SHOT ---->|
                   test eval    |
                               [Metrics: F0.5, AUC, P, R]
```

---

## 5. Comparación con métodos relacionados

### vs Isolation Forest

| Criterio | LPI | Isolation Forest |
|---|---|---|
| Modo | Semi-supervised (usa labels para enrichment) | Unsupervised (no labels) |
| Supuesto | Clase rara concentrada en clusters densos | Clase rara en regiones de baja densidad |
| Output | Score continuo en [0,1] con semántica de probabilidad | Score de anomalía sin semántica de probabilidad |
| Robustez a clustering | Robusta si K* es adecuado | Depende de n_estimators y max_features |
| OPS-SAT-AD F0.5 | 0.670 | 0.381 |

### vs One-Class SVM

| Criterio | LPI | OneClassSVM |
|---|---|---|
| Modo | Semi-supervised | Unsupervised (solo datos normales) |
| Supuesto | Densidad del espacio de features | Frontera maximal de alta densidad |
| Kernel | — | RBF (no lineal) |
| Escalabilidad | O(N log N) GMM | O(N^2) kernel |
| OPS-SAT-AD F0.5 | 0.670 | 0.669 |
| OPS-SAT-AD AUC | 0.920 | 0.800 |

**LPI empata OCSVM en F0.5 pero supera +0.120 en AUC.** Esto significa que LPI produce mejores rankings, lo que tiene valor para priorizar revisión de alertas.

### vs DAGMM (Zong et al. 2018)

| Criterio | LPI | DAGMM |
|---|---|---|
| Representación | Features estadísticas directas | Autoencoder (end-to-end) + GMM en espacio latente |
| Entrenamiento | Separado (features → GMM) | Joint loss (reconstrucción + GMM) |
| Labels | Necesita labels (enrichment) | Unsupervised |
| Complejidad | Baja (~45k params en NF, ~4k en v1) | Alta (~500k típico) |
| [PENDIENTE: resultados DAGMM sobre OPS-SAT-AD] | — | — |

---

## 6. Validación cruzada — protocolo anti-snooping

Para seleccionar el umbral $\tau$ sin contaminar el test set, se usa 5-fold CV con GroupKFold (grupos = segmentos originales, para evitar que ventanas del mismo segmento aparezcan en train y val):

1. Para cada fold: entrenar LPI en K-1 folds, obtener scores OOF en fold restante.
2. Sweep de percentiles sobre scores OOF → $\tau^*$.
3. Entrenar LPI final sobre **todos** los datos de train.
4. Evaluar sobre test exactamente una vez con $\tau^*$.

Este protocolo asegura que el test set nunca informa la selección de hiperparámetros.

Implementado en `src/models/lpi.py:209` (`fit_predict_cv`) y `experiments/s2_lpi/run_lpi_opssat.py`.
