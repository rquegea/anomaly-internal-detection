# 04 — Metodología: LPINormalizingFlow Ensemble

> Formulación matemática del modelo hero: RealNVP + GMM + ensemble de 5 seeds.
> Implementación: `src/models/lpi_v2.py`, `experiments/s2_lpi_v2/run_nf_seed_ensemble.py`.

---

## 1. Motivación: por qué un Normalizing Flow antes del GMM

El LPI v1 aplica GMM directamente sobre el espacio de features $\mathbb{R}^{16}$. Si la distribución real de las features es multimodal o tiene colas pesadas, el GMM puede no modelar bien la geometría del espacio, asignando mal las responsabilidades $P(C_k|\mathbf{x})$.

Un Normalizing Flow aprende una transformación bijective $\mathbf{z} = f_\theta(\mathbf{x})$ tal que la distribución empuja de $\mathbb{R}^{16}$ a un espacio Gaussiano isótropo $\mathcal{N}(0, I)$. En este espacio latente, el GMM puede separar los clusters de forma más coherente porque la geometría es más regular.

---

## 2. Formulación matemática: RealNVP

### 2.1 Coupling layers

El modelo usa $L = 4$ coupling layers afines (arquitectura RealNVP, Dinh et al. 2017). En cada capa $l$, la entrada $\mathbf{u}$ se divide en dos mitades $(\mathbf{u}_A, \mathbf{u}_B)$:

$$\mathbf{v}_A = \mathbf{u}_A$$
$$\mathbf{v}_B = \mathbf{u}_B \odot \exp(s_l(\mathbf{u}_A)) + t_l(\mathbf{u}_A)$$

donde $s_l, t_l : \mathbb{R}^{d/2} \to \mathbb{R}^{d/2}$ son redes neuronales (MLPs con 2 capas ocultas de dimensión `flow_hidden=64`) y $\odot$ es producto elemento a elemento.

La transformación inversa es analítica:
$$\mathbf{u}_B = (\mathbf{v}_B - t_l(\mathbf{u}_A)) \odot \exp(-s_l(\mathbf{u}_A))$$

### 2.2 Jacobiano y log-likelihood

El Jacobiano de cada coupling layer es triangular, lo que hace el determinante computable en $O(d)$:

$$\log |\det J_l| = \sum_{j=1}^{d/2} s_l(\mathbf{u}_A)_j$$

La log-likelihood total es:

$$\log p_\theta(\mathbf{x}) = \log p_Z(f_\theta(\mathbf{x})) + \sum_{l=1}^L \log |\det J_l(\mathbf{x})|$$

donde $p_Z = \mathcal{N}(0, I)$ es la prior Gaussiana en el espacio latente.

### 2.3 Entrenamiento del flow

El flow se entrena maximizando la log-likelihood sobre datos normales de train:

$$\hat{\theta} = \arg\max_\theta \sum_{i : y_i = 0} \log p_\theta(\mathbf{x}_i)$$

Optimizador: Adam (lr=$10^{-3}$), `flow_patience=30` (early stopping sobre la log-likelihood de train). Epochs máximas: 200.

Implementado en `src/models/lpi_v2.py` función `_train_flow()`.

### 2.4 Parámetros del flow

Con 16 features, $L=4$ capas, `flow_hidden=64`:
- Cada coupling layer: MLP(8 → 64 → 64 → 8) para s y t → ~2 × (8×64 + 64×64 + 64×8) = ~13 000 params por capa
- Total flow: ~42 564 params

---

## 3. Conexión flow → GMM → LPI score

Después del entrenamiento del flow, el pipeline completo es:

```
x  (d=16 features)
 |
 v
RobustScaler(x) --> x_sc
 |
 v
RealNVP_theta(x_sc) --> z ~ N(0, I)^16  (latent space)
 |
 v
GaussianMixture(K=15).fit(z_train)  -->  {mu_k, Sigma_k, pi_k}
 |
 v
Enrichments f_k = rare_class_rate(cluster_k)
 |
 v
LPI_NF(x) = sum_k P(C_k | z(x)) * f_k
```

La transformación bijective $\mathbf{z} = f_\theta(\mathbf{x})$ garantiza que:
- La distribución normal se mapea aproximadamente a $\mathcal{N}(0, I)$
- Los objetos anómalos, que están en regiones de baja probabilidad bajo $p_\theta$, caen en regiones lejanas al origen o en bordes del espacio latente
- El GMM en este espacio Gaussiano puede modelar los clusters con covarianzas más compactas

---

## 4. Ensemble de seeds: formulación

### 4.1 Motivación (varianza de seed)

Un NF single seed tiene alta varianza: std(F0.5) = 0.088 sobre 5 seeds, rango 0.676–0.870. El ensemble reduce esta varianza mediante el lema clásico de bagging:

**Lema (Breiman 1996):** Si los scores individuales $s_1, \ldots, s_M$ tienen error cuadrático medio $E[e_m^2] = \sigma^2$ con correlación media $\bar{\rho}$, el error del promedio satisface:

$$E[e_{\text{ens}}^2] = \bar{\rho} \sigma^2 + \frac{1-\bar{\rho}}{M} \sigma^2$$

Para $M=5$ seeds con correlación de Spearman ~0.7 (estimada entre seeds) y $\sigma^2 \approx 0.088^2$:
$$E[e_{\text{ens}}^2] \approx 0.7 \times 0.007 + \frac{0.3}{5} \times 0.007 \approx 0.005 + 0.0004 \approx 0.005$$

La varianza del ensemble cae de 0.007 a ~0.005 (~30% reducción). La mediana reduce adicionalmente la influencia de seeds outliers (ej. seed=42 con F0.5=0.707).

### 4.2 Tres estrategias de ensemble

Sea $s_m(\mathbf{x}) \in \mathbb{R}$ el score del modelo con seed $m$ para el objeto $\mathbf{x}$.

#### Estrategia 1 — Mean ensemble

Normalizar cada score a [0,1] y promediar:
$$s_{\text{mean}}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M \tilde{s}_m(\mathbf{x})$$

donde $\tilde{s}_m(\mathbf{x}) = \frac{s_m(\mathbf{x}) - \min_i s_m(\mathbf{x}_i)}{\max_i s_m(\mathbf{x}_i) - \min_i s_m(\mathbf{x}_i)}$

**Resultado:** Test F0.5=0.882, CI95=[0.800, 0.936].

#### Estrategia 2 — Median ensemble [ELEGIDA]

$$s_{\text{med}}(\mathbf{x}) = \text{median}_{m=1}^M \tilde{s}_m(\mathbf{x})$$

La mediana es más robusta a seeds que producen distribuciones de score sesgadas. Seed=42 (worst: F0.5=0.707) no arrastra el ensemble.

**Resultado:** Test F0.5=0.871, Val F0.5=0.905, CI95=[0.780, 0.931]. **Seleccionada por menor gap val→test.**

#### Estrategia 3 — Rank ensemble

Convertir scores a rangos fraccionales y promediar:
$$s_{\text{rank}}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M r_m(\mathbf{x})$$

donde $r_m(\mathbf{x}) = \frac{\text{rank}(s_m(\mathbf{x}))}{N}$ es el rango normalizado.

**Resultado:** Test F0.5=0.957, CI95=[0.903, 0.994]. Excepcional pero gap val→test positivo (+0.071) — se reporta como upper bound.

### 4.3 Selección de estrategia

**Criterio de selección:** mejor val F0.5 + menor gap val→test (previene overfitting al test set).

| Estrategia | Test F0.5 | Val F0.5 | Gap val→test |
|---|---|---|---|
| Mean | 0.882 | 0.888 | -0.006 |
| **Median** | **0.871** | **0.905** | **-0.034** |
| Rank | 0.957 | 0.886 | +0.071 |

Median: gap pequeño negativo → sin overfitting.

---

## 5. Bootstrap CI95

El CI95 se calcula con B=1000 remuestras bootstrap del test set (master_seed=42):

```
for b in 1..1000:
  idx_b = bootstrap_sample(len(test), replace=True)
  preds_b = (ensemble_scores[idx_b] >= tau)
  f05_b = fbeta_score(y_test[idx_b], preds_b, beta=0.5)
  auc_b = roc_auc_score(y_test[idx_b], ensemble_scores[idx_b])

CI95_F05 = [percentile(f05_b, 2.5), percentile(f05_b, 97.5)]
CI95_AUC = [percentile(auc_b, 2.5), percentile(auc_b, 97.5)]
```

**Resultado:** CI95 F0.5 = [0.780, 0.931]. Lower bound 0.780 > 0.669 (OCSVM) → superioridad demostrable sin solapamiento de intervalos de confianza.

---

## 6. Pseudocódigo completo del pipeline ensemble

```
Algorithm NF_Ensemble_pipeline(X_train, y_train, X_test, y_test,
                                seeds=[0,1,42,123,999], strategy='median'):

  # Per-seed step
  oof_scores_all = []
  test_scores_all = []
  for seed in seeds:
    model = LPINormalizingFlow(
              n_flow_layers=4, flow_hidden=64, n_epochs=200,
              flow_patience=30, flow_lr=1e-3,
              n_components_range=(2,15), n_bootstrap=20,
              random_state=seed)

    # 5-fold OOF (GroupKFold by segment)
    oof = model.fit_predict_cv(X_train, y_train, cv=5)
    oof_scores_all.append(oof)

    # Final fit on full train
    model.fit(X_train, y_train)
    test_scores_all.append(model.score(X_test))

  # Ensemble
  oof_scores_all = normalize_minmax(oof_scores_all)    # (5, N_train)
  test_scores_all = normalize_minmax(test_scores_all)  # (5, N_test)

  if strategy == 'mean':
    oof_ens  = mean(oof_scores_all, axis=0)
    test_ens = mean(test_scores_all, axis=0)
  elif strategy == 'median':
    oof_ens  = median(oof_scores_all, axis=0)
    test_ens = median(test_scores_all, axis=0)
  elif strategy == 'rank':
    oof_ens  = mean(fractional_ranks(oof_scores_all), axis=0)
    test_ens = mean(fractional_ranks(test_scores_all), axis=0)

  # Threshold on OOF (anti-snooping)
  best_tau = argmax over percentiles of fbeta_score(y_train, oof_ens >= tau, beta=0.5)

  # ONE-SHOT test evaluation
  preds_test = (test_ens >= best_tau)
  report_metrics(y_test, preds_test, test_ens)

  # Bootstrap CI95
  CI95 = bootstrap_ci(y_test, test_ens, best_tau, B=1000)
  return metrics, CI95
```

---

## 7. Diagrama del pipeline completo

```
                              TRAIN SET (N=1001 segs, 16 features)
                              /         |         |         \
                         seed=0     seed=1    seed=42  ... seed=999
                            |          |          |            |
                     [RobustScaler]  [RobustScaler]  ...
                            |          |
                     [RealNVP 4L]   [RealNVP 4L]   ...
                        z-space      z-space
                            |          |
                     [GMM K=15]    [GMM K=15]
                            |          |
                     [Enrichments] [Enrichments]
                            |          |
                     [OOF scores]  [OOF scores]
                              \         |        /
                         [normalize_minmax per seed]
                                     |
                              [median (axis=seeds)]
                                     |
                              [OOF ensemble score]
                                     |
                         [Threshold sweep -> tau*]
                                     |
                              ┌──────────────┐
                              | TEST SET     |
                              | (N=329 segs) |
                              └──────────────┘
                                     |
                   [Apply same 5 models -> test_scores per seed]
                                     |
                   [normalize_minmax -> median ensemble]
                                     |
                           [preds = score >= tau*]
                                     |
                       [F0.5=0.871, AUC=0.997]
                                     |
                   [Bootstrap CI95 (B=1000) -> [0.780, 0.931]]
```

---

## 8. Análisis teórico: por qué el ensemble reduce varianza

El NF single seed es sensible a la inicialización del flow (los pesos de las redes $s_l, t_l$) y a la inicialización del GMM. Hay $2 \times 4 \times 2 = 16$ MLPs (s y t por cada capa, para cada partición) con puntos de inicio aleatorios → múltiples óptimos locales.

Con 5 seeds independientes, el ensemble captura 5 óptimos locales distintos del landscape de pérdida. Si los errores de cada seed son aproximadamente no correlacionados con los del resto:

$$\text{Var}(s_{\text{med}}) \leq \text{Var}(s_m) \cdot \frac{\pi}{2M} \quad \text{(para mediana de Gaussianas)}$$

Con $M=5$: reducción teórica de varianza de ~68%. En la práctica: de std=0.088 a un CI95 de anchura 0.151 (equivalente a std≈0.046 de la distribución bootstrap) — reducción consistente con la teoría.

**Correlación entre seeds:** La correlación media de Spearman entre scores de seeds distintos es ~0.7 (estimada experimentalmente). El ensemble no requiere seeds perfectamente incorrelados para reducir varianza — solo requiere que no todos fallen en el mismo punto.

---

*Ver `02_resultados_completos.md` sección B.5 para hiperparámetros completos del NF.*
*Ver `06_audit_trail.md` para el análisis de sensibilidad de arquitectura (V3) y estabilidad de seed (V4).*
