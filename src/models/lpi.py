"""
Latent Propensity Index (LPI) for anomaly detection in satellite telemetry.

Semi-supervised algorithm that detects rare-class enrichment in GMM clusters.

Algorithm (Quesada 2026):
    1. Standardize features with RobustScaler.
    2. Select K Gaussian components by BIC averaged over bootstrap resamples
       (K in n_components_range, n_bootstrap resamples of 80% of the data).
    3. Fit final GMM with the selected K.
    4. Compute cluster enrichment:
           f_k = fraction of rare-class samples hard-assigned to cluster k
    5. LPI score: LPI(x) = sum_k P(C_k | x) * f_k

Semi-supervised: fit() requires labels to compute enrichment f_k.
Labels are never used for the GMM geometry — only to measure rare-class
concentration per cluster after fitting.

Published validations (Quesada 2026):
    - CHIME/FRB Catalog 2 (N=3641, 83 repeaters): AUC=0.760±0.048
    - eROSITA dwarf galaxies (N=169, 95 IMBHs): AUC=0.708±0.103
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class LPIDetector:
    """
    Latent Propensity Index detector.

    Parameters
    ----------
    n_components_range : tuple[int, int]
        Min and max number of GMM components to search (inclusive).
    n_bootstrap : int
        Number of bootstrap resamples for BIC averaging.
        Each resample uses 80% of training data sampled with replacement.
    scaler : {'robust', 'standard'}
        Feature scaling applied before GMM. 'robust' recommended for
        telemetry data with outlier features.
    random_state : int
        Master random seed. Propagated to all internal stochastic steps.
    """

    def __init__(
        self,
        n_components_range: tuple[int, int] = (2, 15),
        n_bootstrap: int = 20,
        scaler: str = "robust",
        random_state: int = 42,
    ) -> None:
        self.n_components_range = n_components_range
        self.n_bootstrap = n_bootstrap
        self.scaler = scaler
        self.random_state = random_state

        # Fitted state — None until fit() is called
        self._scaler: RobustScaler | StandardScaler | None = None
        self._gmm: GaussianMixture | None = None
        self._enrichments: np.ndarray | None = None
        self._best_k: int | None = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_scaler(self) -> RobustScaler | StandardScaler:
        if self.scaler == "robust":
            return RobustScaler()
        return StandardScaler()

    def _select_k_by_bic(
        self, X_scaled: np.ndarray
    ) -> tuple[int, dict[int, float]]:
        """
        Select K by averaging BIC over bootstrap resamples.

        For each of n_bootstrap iterations:
            - Resample 80% of X_scaled with replacement
            - Fit a GMM for every K in the search range
            - Record BIC on that resample

        Returns the K with the lowest mean BIC across all bootstraps.
        """
        rng = np.random.RandomState(self.random_state)
        k_min, k_max = self.n_components_range
        ks = list(range(k_min, k_max + 1))

        bic_accumulator: dict[int, list[float]] = {k: [] for k in ks}

        n = len(X_scaled)
        # Guard against tiny datasets (used in unit tests)
        boot_size = max(int(0.8 * n), 2 * k_max)
        boot_size = min(boot_size, n)  # can't exceed dataset size

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=boot_size, replace=True)
            X_boot = X_scaled[idx]
            seed_b = int(rng.randint(0, 100_000))

            for k in ks:
                # Skip K larger than unique samples (can happen in tiny boots)
                if k >= len(np.unique(idx)):
                    bic_accumulator[k].append(np.inf)
                    continue
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=seed_b,
                    max_iter=200,
                    n_init=1,
                    reg_covar=1e-6,
                )
                try:
                    gmm.fit(X_boot)
                    bic_accumulator[k].append(gmm.bic(X_boot))
                except Exception:
                    bic_accumulator[k].append(np.inf)

        mean_bic = {k: float(np.mean(v)) for k, v in bic_accumulator.items()}
        best_k = min(mean_bic, key=mean_bic.get)
        return best_k, mean_bic

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LPIDetector":
        """
        Fit the LPI detector.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix — one row per object.
        y : array-like, shape (n_samples,)
            Binary labels where 1 marks the rare class (anomaly).
            Used only for computing cluster enrichment, not for GMM geometry.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        # Scale features
        self._scaler = self._make_scaler()
        X_scaled = self._scaler.fit_transform(X)

        # Select K via BIC bootstrap
        self._best_k, mean_bic = self._select_k_by_bic(X_scaled)
        logger.info(
            "LPI: BIC-selected K=%d  (mean BIC: %s)",
            self._best_k,
            {k: f"{v:.1f}" for k, v in mean_bic.items()},
        )
        print(f"[LPI] BIC-selected K={self._best_k}")

        # Fit final GMM with more restarts for stability
        self._gmm = GaussianMixture(
            n_components=self._best_k,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=300,
            n_init=5,
            reg_covar=1e-6,
        )
        self._gmm.fit(X_scaled)

        # Compute enrichment: f_k = rare-class rate among hard-assigned members
        hard_assignments = self._gmm.predict(X_scaled)  # (n_samples,)
        self._enrichments = np.zeros(self._best_k, dtype=float)
        for k in range(self._best_k):
            mask = hard_assignments == k
            if mask.sum() > 0:
                self._enrichments[k] = float(y[mask].mean())

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample LPI scores.

        LPI(x) = sum_k P(C_k | x) * f_k

        Higher score indicates higher propensity for the rare class.

        Returns
        -------
        scores : (n_samples,) float array in [0, 1]
        """
        if self._gmm is None:
            raise RuntimeError("Call fit() before score().")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        responsibilities = self._gmm.predict_proba(X_scaled)  # (n_samples, K)
        return responsibilities @ self._enrichments              # (n_samples,)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Binarize LPI scores: 1 (anomaly) if score >= threshold, else 0.
        """
        return (self.score(X) >= threshold).astype(int)

    def fit_predict_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> np.ndarray:
        """
        KFold cross-validation returning out-of-fold LPI scores.

        Fits a fresh LPIDetector on each training fold (using only the labels
        in that fold) and scores the held-out validation fold. No data from
        the validation fold leaks into fit().

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,) binary labels
        cv : number of folds

        Returns
        -------
        oof_scores : (n_samples,) out-of-fold LPI scores
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        oof_scores = np.zeros(len(X), dtype=float)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Each fold gets a different seed to avoid identical BIC trajectories
            fold_detector = LPIDetector(
                n_components_range=self.n_components_range,
                n_bootstrap=self.n_bootstrap,
                scaler=self.scaler,
                random_state=self.random_state + fold_idx + 1,
            )
            fold_detector.fit(X[train_idx], y[train_idx])
            oof_scores[val_idx] = fold_detector.score(X[val_idx])
            print(
                f"  [LPI] Fold {fold_idx + 1}/{cv} done — "
                f"K={fold_detector._best_k}  "
                f"enrichments={fold_detector._enrichments.round(3).tolist()}"
            )

        return oof_scores

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def best_k(self) -> int | None:
        """K selected by BIC (available after fit())."""
        return self._best_k

    @property
    def enrichments(self) -> np.ndarray | None:
        """Per-cluster rare-class enrichment f_k (available after fit())."""
        return self._enrichments
