# 08 — Bibliografía: Papers a Citar

> Referencias organizadas por tema para Paper 1 (LPI original) y Paper 3 (NF ensemble).
> Formato: (Autor Año) en el texto, BibTeX completo aquí.
> Mínimo 30 referencias entre los dos papers.

---

## Paper 1 — LPI original: CHIME/FRB + eROSITA para MNRAS Letters

### P1.1 — Datasets astrofísicos

```bibtex
@article{chime2023catalog2,
  title        = {[VERIFICAR: título exacto del CHIME/FRB Catalog 2]},
  author       = {CHIME/FRB Collaboration},
  journal      = {[VERIFICAR: journal — posiblemente ApJS o arXiv]},
  year         = {2023},
  doi          = {[VERIFICAR: DOI del catálogo]},
  note         = {Catalog 2 de FRBs detectados por CHIME. N=3641 eventos, 83 repeaters.}
}

@article{erosita2024dwarfs,
  title        = {[VERIFICAR: título exacto del paper eROSITA dwarf galaxies]},
  author       = {[VERIFICAR: autores]},
  journal      = {[VERIFICAR: journal]},
  year         = {2024},
  doi          = {[VERIFICAR: DOI]},
  note         = {Catálogo de galaxias enanas con candidatas a IMBH detectadas por eROSITA. N=169.}
}
```

### P1.2 — FRB repeaters: contexto científico

```bibtex
@article{petroff2022frb,
  title        = {Fast Radio Bursts},
  author       = {Petroff, E. and Hessels, J.W.T. and Lorimer, D.R.},
  journal      = {The Astronomy and Astrophysics Review},
  volume       = {30},
  pages        = {2},
  year         = {2022},
  doi          = {10.1007/s00159-022-00139-w}
}

@article{CHIME2021catalog1,
  title        = {The First CHIME/FRB Fast Radio Burst Catalog},
  author       = {CHIME/FRB Collaboration},
  journal      = {The Astrophysical Journal Supplement Series},
  volume       = {257},
  number       = {2},
  pages        = {59},
  year         = {2021},
  doi          = {10.3847/1538-4365/ac33ab}
}
```

### P1.3 — Métodos estadísticos base del LPI

```bibtex
@book{bishop2006prml,
  title        = {Pattern Recognition and Machine Learning},
  author       = {Bishop, Christopher M.},
  publisher    = {Springer},
  year         = {2006},
  note         = {Referencia canónica para Gaussian Mixture Models (Capítulo 9) y EM algorithm.}
}

@article{schwarz1978bic,
  title        = {Estimating the dimension of a model},
  author       = {Schwarz, Gideon},
  journal      = {The Annals of Statistics},
  volume       = {6},
  number       = {2},
  pages        = {461--464},
  year         = {1978},
  doi          = {10.1214/aos/1176344136},
  note         = {Paper original del BIC (Bayesian Information Criterion).}
}

@article{efron1979bootstrap,
  title        = {Bootstrap Methods: Another Look at the Jackknife},
  author       = {Efron, Bradley},
  journal      = {The Annals of Statistics},
  volume       = {7},
  number       = {1},
  pages        = {1--26},
  year         = {1979},
  doi          = {10.1214/aos/1176344552},
  note         = {Paper original del bootstrap estadístico.}
}

@book{efron1994bootstrap,
  title        = {An Introduction to the Bootstrap},
  author       = {Efron, Bradley and Tibshirani, Robert J.},
  publisher    = {Chapman and Hall/CRC},
  year         = {1994},
  doi          = {10.1201/9780429246593}
}
```

### P1.4 — Anomaly detection general (contexto)

```bibtex
@article{chandola2009anomaly,
  title        = {Anomaly Detection: A Survey},
  author       = {Chandola, Varun and Banerjee, Arindam and Kumar, Vipin},
  journal      = {ACM Computing Surveys},
  volume       = {41},
  number       = {3},
  pages        = {15},
  year         = {2009},
  doi          = {10.1145/1541880.1541882}
}

@article{liu2008isolation,
  title        = {Isolation Forest},
  author       = {Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  booktitle    = {2008 Eighth IEEE International Conference on Data Mining},
  pages        = {413--422},
  year         = {2008},
  doi          = {10.1109/ICDM.2008.17}
}

@article{scholkopf2001ocsvm,
  title        = {Estimating the Support of a High-Dimensional Distribution},
  author       = {Sch\"{o}lkopf, Bernhard and Platt, John C. and Shawe-Taylor, John and Smola, Alexander J. and Williamson, Robert C.},
  journal      = {Neural Computation},
  volume       = {13},
  number       = {7},
  pages        = {1443--1471},
  year         = {2001},
  doi          = {10.1162/089976601750264965}
}
```

---

## Paper 3 — LPINormalizingFlow ensemble: OPS-SAT-AD para NeurIPS ML4PS

### P3.1 — Dataset OPS-SAT-AD

```bibtex
@article{ruszczak2025opssat,
  title        = {The OPS-SAT benchmark for detecting anomalies in satellite telemetry},
  author       = {Ruszczak, Bogdan and Kotowski, Krzysztof and Evans, Dave and Nalepa, Jakub},
  journal      = {Scientific Data},
  publisher    = {Springer Nature},
  year         = {2025},
  doi          = {10.1038/s41597-025-05035-3},
  note         = {Paper principal del benchmark OPS-SAT-AD con 30 baselines. Nature Scientific Data.}
}

@misc{ruszczak2024opssatzenodo,
  title        = {{OPSSAT-AD} -- anomaly detection dataset for satellite telemetry},
  author       = {Ruszczak, Bogdan and Kotowski, Krzysztof and Evans, Dave and Nalepa, Jakub},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12588359},
  note         = {Dataset v1. Versión actualizada abril 2025: https://zenodo.org/records/15108715}
}

@inproceedings{ruszczak2023iccs,
  title        = {Machine Learning Detects Anomalies in {OPS-SAT} Telemetry},
  author       = {Ruszczak, Bogdan and Kotowski, Krzysztof and Andrzejewski, Julian and others},
  booktitle    = {Computational Science -- ICCS 2023},
  series       = {Lecture Notes in Computer Science},
  volume       = {14073},
  publisher    = {Springer, Cham},
  year         = {2023},
  doi          = {10.1007/978-3-031-35995-8_21}
}
```

### P3.2 — Normalizing Flows (método principal)

```bibtex
@inproceedings{dinh2017realnvp,
  title        = {Density Estimation Using Real-Valued Non-Volume Preserving ({Real-NVP}) Transformations},
  author       = {Dinh, Laurent and Sohl-Dickstein, Jascha and Bengio, Samy},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2017},
  url          = {https://arxiv.org/abs/1605.08803},
  note         = {Paper original de RealNVP -- arquitectura usada en este trabajo.}
}

@article{papamakarios2021normalizing,
  title        = {Normalizing Flows for Probabilistic Modeling and Inference},
  author       = {Papamakarios, George and Nalisnick, Eric and Rezende, Danilo Jimenez and Mohamed, Shakir and Lakshminarayanan, Balaji},
  journal      = {Journal of Machine Learning Research},
  volume       = {22},
  number       = {57},
  pages        = {1--64},
  year         = {2021},
  url          = {https://jmlr.org/papers/v22/19-1028.html}
}

@inproceedings{rezende2015nf,
  title        = {Variational Inference with Normalizing Flows},
  author       = {Rezende, Danilo Jimenez and Mohamed, Shakir},
  booktitle    = {International Conference on Machine Learning (ICML)},
  pages        = {1530--1538},
  year         = {2015},
  url          = {https://arxiv.org/abs/1505.05770}
}

@inproceedings{dinh2015nice,
  title        = {{NICE}: Non-linear Independent Components Estimation},
  author       = {Dinh, Laurent and Krueger, David and Bengio, Yoshua},
  booktitle    = {ICLR Workshop},
  year         = {2015},
  url          = {https://arxiv.org/abs/1410.8516}
}
```

### P3.3 — Anomaly detection con Normalizing Flows

```bibtex
@article{osada2023nf_anomaly,
  title        = {Unsupervised Anomaly Detection Using Normalizing Flows},
  author       = {Osada, Genki and Ahmedt-Aristizabal, David and Gedeon, Tom},
  journal      = {Neurocomputing},
  year         = {2023},
  doi          = {[VERIFICAR: DOI exacto del paper Osada 2023 Neurocomputing]},
  note         = {Uso de NF para anomaly detection unsupervised -- referencia directa.}
}

@article{nalisnick2019detecting,
  title        = {Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality},
  author       = {Nalisnick, Eric and Matsukawa, Akihiro and Teh, Yee Whye and Lakshminarayanan, Balaji},
  journal      = {arXiv preprint arXiv:1906.02994},
  year         = {2019},
  url          = {https://arxiv.org/abs/1906.02994}
}
```

### P3.4 — Anomaly detection en telemetría satelital

```bibtex
@article{gonzalez2025transformers,
  title        = {[VERIFICAR: título exacto del paper Gonzalez 2025]},
  author       = {Gonzalez, [et al.]},
  journal      = {Acta Astronautica},
  year         = {2025},
  doi          = {[identificador interno S0094576525006095 en ScienceDirect]},
  url          = {https://www.sciencedirect.com/science/article/pii/S0094576525006095},
  note         = {Transformers superando LSTM y otros baselines en OPS-SAT-AD. Referencia directa más relevante.}
}

@article{hundman2018detecting,
  title        = {Detecting Spacecraft Anomalies Using {LSTMs} and Nonparametric Dynamic Thresholding},
  author       = {Hundman, Kyle and Constantinou, Valentino and Laporte, Christopher and Colwell, Ian and Soderstrom, Tom},
  booktitle    = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages        = {387--395},
  year         = {2018},
  doi          = {10.1145/3219819.3219845},
  note         = {LSTM-AE para anomaly detection en telemetría SMAP/MSL de NASA.}
}

@article{su2019robust,
  title        = {Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network},
  author       = {Su, Ya and Zhao, Youjian and Niu, Chenhao and Liu, Rong and Sun, Wei and Pei, Dan},
  booktitle    = {Proceedings of the 25th ACM SIGKDD},
  pages        = {2828--2837},
  year         = {2019},
  doi          = {10.1145/3292500.3330672}
}
```

### P3.5 — Métodos deep para anomaly detection (competidores)

```bibtex
@inproceedings{zong2018dagmm,
  title        = {Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection},
  author       = {Zong, Bo and Song, Qi and Min, Martin Renqiang and Cheng, Wei and Lumezanu, Cristian and Cho, Daeki and Chen, Haifeng},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2018},
  url          = {https://openreview.net/forum?id=BJJLHbb0-}
}

@inproceedings{ruff2018deepsvdd,
  title        = {Deep One-Class Classification},
  author       = {Ruff, Lukas and Vandermeulen, Robert A. and G\"{o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib Ahmed and Binder, Alexander and M\"{u}ller, Emmanuel and Kloft, Marius},
  booktitle    = {International Conference on Machine Learning (ICML)},
  pages        = {4393--4402},
  year         = {2018},
  url          = {https://arxiv.org/abs/1801.05188}
}

@inproceedings{xu2022anomaly_transformer,
  title        = {Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
  author       = {Xu, Jiehui and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2022},
  url          = {https://arxiv.org/abs/2110.02642}
}
```

### P3.6 — Transformers y series temporales

```bibtex
@inproceedings{vaswani2017attention,
  title        = {Attention Is All You Need},
  author       = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume       = {30},
  year         = {2017},
  url          = {https://arxiv.org/abs/1706.03762}
}

@inproceedings{nie2023patchtst,
  title        = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author       = {Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2023},
  url          = {https://arxiv.org/abs/2211.14730},
  note         = {PatchTST — mencionado como extensión opcional en S3.}
}
```

### P3.7 — Ensemble methods y bagging

```bibtex
@article{breiman1996bagging,
  title        = {Bagging Predictors},
  author       = {Breiman, Leo},
  journal      = {Machine Learning},
  volume       = {24},
  number       = {2},
  pages        = {123--140},
  year         = {1996},
  doi          = {10.1007/BF00058655},
  note         = {Paper original de bagging. Lema de reducción de varianza mediante averaging.}
}

@article{dietterich2000ensemble,
  title        = {Ensemble Methods in Machine Learning},
  author       = {Dietterich, Thomas G.},
  booktitle    = {International Workshop on Multiple Classifier Systems},
  pages        = {1--15},
  year         = {2000},
  doi          = {10.1007/3-540-45014-9_1}
}
```

### P3.8 — Variational inference y BayesianGMM

```bibtex
@article{blei2017variational,
  title        = {Variational Inference: A Review for Statisticians},
  author       = {Blei, David M. and Kucukelbir, Alp and McAuliffe, Jon D.},
  journal      = {Journal of the American Statistical Association},
  volume       = {112},
  number       = {518},
  pages        = {859--877},
  year         = {2017},
  doi          = {10.1080/01621459.2017.1285773}
}

@article{blei2006variational_dp,
  title        = {Variational Inference for Dirichlet Process Mixtures},
  author       = {Blei, David M. and Jordan, Michael I.},
  journal      = {Bayesian Analysis},
  volume       = {1},
  number       = {1},
  pages        = {121--143},
  year         = {2006},
  doi          = {10.1214/06-BA104}
}
```

### P3.9 — Métricas de evaluación

```bibtex
@article{fawcett2006roc,
  title        = {An Introduction to {ROC} Analysis},
  author       = {Fawcett, Tom},
  journal      = {Pattern Recognition Letters},
  volume       = {27},
  number       = {8},
  pages        = {861--874},
  year         = {2006},
  doi          = {10.1016/j.patrec.2005.10.010}
}

@article{powers2011evaluation,
  title        = {Evaluation: From Precision, Recall and {F-Measure} to {ROC}, Informedness, Markedness and Correlation},
  author       = {Powers, David Martin},
  journal      = {Journal of Machine Learning Technologies},
  volume       = {2},
  number       = {1},
  pages        = {37--63},
  year         = {2011}
}
```

---

## Conteo total de referencias

| Paper | Referencias | Estado |
|---|---|---|
| Paper 1 | ~12 referencias | 4 requieren [VERIFICAR] |
| Paper 3 | ~20 referencias | 3 requieren [VERIFICAR] |
| Compartidas | ~3 | — |
| **Total único** | **~29** | ~7 [VERIFICAR] pendientes |

> Nota: El objetivo de 30+ referencias está cubierto. Los campos [VERIFICAR] corresponden a papers cuyos DOIs o títulos exactos no están en el CLAUDE.md y necesitan confirmación de Rodrigo.
