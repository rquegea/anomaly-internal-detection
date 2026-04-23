"""
LPI v2 — Deep-tech extensions of the Latent Propensity Index (Quesada 2026).

Five algorithmic extensions that move LPI from "applied AI with known techniques"
to "mathematically novel and publishable at NeurIPS / ICML level":

    1. LPINormalizingFlow  — RealNVP replaces GMM density in feature space.
                             Captures non-Gaussian, multi-modal structure that
                             makes GMM components incoherent in the original space.

    2. LPIVariational      — sklearn BayesianGaussianMixture replaces standard GMM.
                             Variational inference handles K selection automatically;
                             soft posterior assignments give richer enrichment signal.

    3. LPIBayesian         — Bootstrap uncertainty over the enrichment f_k.
                             Every LPI score comes with a 90 % confidence interval —
                             critical for operational systems where "I don't know"
                             is safer than a false negative.

    4. LPIHierarchical     — Two-level GMM: macro-clusters capture global structure,
                             micro-clusters refine within each macro-cluster.
                             Handles heterogeneous anomaly subtypes gracefully.

    5. LPIOnline           — Streaming / adaptive model: parameters update with each
                             new data batch without full retraining. Addresses real
                             telemetry requirement: anomaly patterns drift over mission
                             life, and competitors' batch models go stale.

All five expose the same interface as LPIDetector (src/models/lpi.py):
    fit(X, y)          → self
    score(X)           → (n_samples,) float in [0, 1]
    predict(X, thr)    → (n_samples,) int {0, 1}
    fit_predict_cv(X, y, cv=5) → (n_samples,) OOF scores

Seed 42 propagated throughout for reproducibility.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.lpi import LPIDetector

logger = logging.getLogger(__name__)

# ── Optional PyTorch (Extension 1 only) ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found — LPINormalizingFlow will raise on use. "
        "Install with: uv add torch",
        stacklevel=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared CV utility
# ─────────────────────────────────────────────────────────────────────────────


def _cv_with_factory(
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    factory,
    class_name: str = "LPI",
    random_state: int = 42,
) -> np.ndarray:
    """
    KFold cross-validation using a detector factory.

    Parameters
    ----------
    factory : callable
        Called as factory(fold_idx) → fitted-able detector.
        Must implement fit(X, y) and score(X).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_scores = np.zeros(len(X), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        det = factory(fold_idx)
        det.fit(X[train_idx], y[train_idx])
        oof_scores[val_idx] = det.score(X[val_idx])
        k_str = str(getattr(det, "_best_k", "?"))
        print(f"  [{class_name}] Fold {fold_idx + 1}/{cv}  K={k_str}")

    return oof_scores


# =============================================================================
# Extension 1 — LPINormalizingFlow
# =============================================================================


class _AffineCouple(nn.Module):
    """
    Affine coupling layer for RealNVP (Dinh et al., 2017).

    Given a split (conditioned, to_transform):
        s = tanh-network(conditioned)
        t = tanh-network(conditioned)
        y = to_transform * exp(s) + t
        log|det J| = sum(s)
    """

    def __init__(self, d_cond: int, d_transform: int, hidden: int = 64) -> None:
        super().__init__()
        self.net_s = nn.Sequential(
            nn.Linear(d_cond, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_transform),
        )
        self.net_t = nn.Sequential(
            nn.Linear(d_cond, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_transform),
        )

    def forward(
        self, cond: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.net_s(cond).clamp(-4.0, 4.0)
        t = self.net_t(cond)
        y = x * torch.exp(s) + t
        return y, s.sum(dim=1)

    def inverse(self, cond: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = self.net_s(cond).clamp(-4.0, 4.0)
        t = self.net_t(cond)
        return (y - t) * torch.exp(-s)


class _RealNVP(nn.Module):
    """
    Lightweight RealNVP for tabular data with alternating coupling splits.

    Layers alternate conditioning on the first vs second half of the input,
    ensuring all dimensions are transformed at least once every two layers.
    """

    def __init__(self, d: int, n_layers: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.d = d
        self.d_half = d // 2
        self.d_rest = d - self.d_half
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                # Condition on [:d_half], transform [d_half:]
                self.layers.append(_AffineCouple(self.d_half, self.d_rest, hidden))
            else:
                # Condition on [d_half:], transform [:d_half]
                self.layers.append(_AffineCouple(self.d_rest, self.d_half, hidden))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x → (z, log_det) where z ≈ N(0, I) after training."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                cond, to_tr = x[:, : self.d_half], x[:, self.d_half :]
                y, ld = layer(cond, to_tr)
                x = torch.cat([cond, y], dim=1)
            else:
                cond, to_tr = x[:, self.d_half :], x[:, : self.d_half]
                y, ld = layer(cond, to_tr)
                x = torch.cat([y, cond], dim=1)
            log_det = log_det + ld
        return x, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log p(x) = log N(z; 0, I) + log|det J|."""
        z, log_det = self.forward(x)
        log_pz = (
            -0.5 * (z**2).sum(dim=1)
            - 0.5 * self.d * np.log(2.0 * np.pi)  # Python float — no device issue
        )
        return log_pz + log_det


def _train_flow(
    flow: _RealNVP,
    X_tensor: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
) -> _RealNVP:
    """Train RealNVP by maximising training log-likelihood with early stopping."""
    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-5
    )
    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in flow.state_dict().items()}
    no_improve = 0

    flow.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = -flow.log_prob(X_tensor).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.detach())

        loss_val = float(loss.item())
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.debug("Flow early-stop epoch=%d  loss=%.4f", epoch, best_loss)
            break

    flow.load_state_dict(best_state)
    flow.eval()
    return flow


class LPINormalizingFlow(LPIDetector):
    """
    LPI with RealNVP Normalizing Flow density estimation (Extension 1).

    Architecture:
        RobustScaler → RealNVP (n_flow_layers coupling layers) → latent Z
        GMM(K, BIC) on Z → enrichment f_k → LPI score

    Key contribution vs v1:
        The bijective mapping X → Z allows GMM to fit in a (approximately)
        Gaussian latent space, resolving cluster incoherence caused by
        non-Gaussian tails and multi-modal feature distributions.

    Reference: Dinh et al. (2017) "Density estimation using Real-NVP".
    """

    def __init__(
        self,
        n_components_range: tuple[int, int] = (2, 15),
        n_bootstrap: int = 20,
        scaler: str = "robust",
        random_state: int = 42,
        n_flow_layers: int = 4,
        flow_hidden: int = 64,
        n_epochs: int = 200,
        flow_lr: float = 1e-3,
        flow_patience: int = 30,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_components_range, n_bootstrap, scaler, random_state)
        self.n_flow_layers = n_flow_layers
        self.flow_hidden = flow_hidden
        self.n_epochs = n_epochs
        self.flow_lr = flow_lr
        self.flow_patience = flow_patience
        self.device = device
        self._flow: Optional[_RealNVP] = None

    def _to_latent(self, X_scaled: np.ndarray) -> np.ndarray:
        dev = torch.device(self.device)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=dev)
        with torch.no_grad():
            Z_t, _ = self._flow(X_t)
        return Z_t.cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LPINormalizingFlow":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for LPINormalizingFlow.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # 1. Scale
        self._scaler = self._make_scaler()
        X_scaled = self._scaler.fit_transform(X)

        # 2. Train RealNVP
        dev = torch.device(self.device)
        torch.manual_seed(self.random_state)
        n_features = X_scaled.shape[1]
        self._flow = _RealNVP(n_features, self.n_flow_layers, self.flow_hidden).to(dev)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=dev)
        self._flow = _train_flow(
            self._flow,
            X_tensor,
            n_epochs=self.n_epochs,
            lr=self.flow_lr,
            patience=self.flow_patience,
        )

        # 3. Latent representation
        Z = self._to_latent(X_scaled)

        # 4. BIC K selection in latent space (reuse parent method)
        self._best_k, mean_bic = self._select_k_by_bic(Z)
        print(f"[LPINormalizingFlow] BIC K={self._best_k}  flow_params={self.n_flow_params}")

        # 5. Final GMM on Z
        self._gmm = GaussianMixture(
            n_components=self._best_k,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=300,
            n_init=5,
            reg_covar=1e-6,
        )
        self._gmm.fit(Z)

        # 6. Enrichments (same formula as v1)
        hard_assignments = self._gmm.predict(Z)
        self._enrichments = np.zeros(self._best_k, dtype=float)
        for k in range(self._best_k):
            mask = hard_assignments == k
            if mask.sum() > 0:
                self._enrichments[k] = float(y[mask].mean())

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._gmm is None:
            raise RuntimeError("Call fit() before score().")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        Z = self._to_latent(X_scaled)
        resp = self._gmm.predict_proba(Z)
        return resp @ self._enrichments

    def _make_clone(self, seed_offset: int) -> "LPINormalizingFlow":
        return LPINormalizingFlow(
            n_components_range=self.n_components_range,
            n_bootstrap=self.n_bootstrap,
            scaler=self.scaler,
            random_state=self.random_state + seed_offset,
            n_flow_layers=self.n_flow_layers,
            flow_hidden=self.flow_hidden,
            n_epochs=self.n_epochs,
            flow_lr=self.flow_lr,
            flow_patience=self.flow_patience,
            device=self.device,
        )

    def fit_predict_cv(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        return _cv_with_factory(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            cv,
            factory=lambda fold_idx: self._make_clone(fold_idx + 1),
            class_name="LPINormalizingFlow",
            random_state=self.random_state,
        )

    @property
    def n_flow_params(self) -> int:
        """Number of trainable parameters in the RealNVP flow."""
        if self._flow is None:
            return 0
        return sum(p.numel() for p in self._flow.parameters())

    @property
    def n_params_effective(self) -> int:
        d = self._gmm.means_.shape[1] if self._gmm else 0
        gmm_params = self._best_k * (d + d * (d + 1) // 2 + 1) if self._best_k else 0
        return self.n_flow_params + gmm_params


# =============================================================================
# Extension 2 — LPIVariational
# =============================================================================


class LPIVariational(LPIDetector):
    """
    LPI with variational Bayes GMM (Extension 2).

    Replaces the standard MLE GMM + BIC bootstrap with sklearn's
    BayesianGaussianMixture, which:
      - Maximises an ELBO (evidence lower bound) instead of likelihood
      - Applies a Dirichlet process prior over mixture weights →
        components with no support automatically shrink to zero weight
      - No separate K-selection step; K_max is an upper bound and
        effective K is determined by the data

    Enrichment modification:
        f_k = sum_i P(k|x_i) * y_i / sum_i P(k|x_i)   (soft-weighted)
    vs v1 hard assignment.  Soft assignments reduce sensitivity to
    boundary ambiguity in overlapping clusters.

    References:
        Blei & Jordan (2006), Variational Inference for Dirichlet Process Mixtures.
        Attias (2000), A Variational Bayesian Framework for Graphical Models.
    """

    def __init__(
        self,
        k_max: int = 15,
        scaler: str = "robust",
        random_state: int = 42,
        weight_concentration_prior: float = 1e-2,
        covariance_type: str = "full",
        n_bootstrap: int = 0,  # unused; kept for interface compatibility
        n_components_range: tuple[int, int] = (2, 15),
    ) -> None:
        super().__init__(n_components_range, n_bootstrap, scaler, random_state)
        self.k_max = k_max
        self.weight_concentration_prior = weight_concentration_prior
        self.covariance_type = covariance_type
        self._bgmm: Optional[BayesianGaussianMixture] = None
        self._effective_k: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LPIVariational":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # 1. Scale
        self._scaler = self._make_scaler()
        X_scaled = self._scaler.fit_transform(X)

        # 2. Fit variational Bayes GMM — no BIC bootstrap needed
        # reg_covar=1e-3 prevents singular precision matrices on small folds
        self._bgmm = BayesianGaussianMixture(
            n_components=self.k_max,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=self.weight_concentration_prior,
            random_state=self.random_state,
            max_iter=500,
            n_init=3,
            reg_covar=1e-3,
        )
        self._bgmm.fit(X_scaled)

        # Effective K: components with mean weight > 1 %
        weights = self._bgmm.weights_
        active = weights > 0.01
        self._effective_k = int(active.sum())
        self._best_k = self._effective_k  # for interface compatibility
        print(
            f"[LPIVariational] effective K={self._effective_k}"
            f"  (K_max={self.k_max}, DP prior={self.weight_concentration_prior})"
        )

        # 3. Soft-weighted enrichment
        #    f_k = E_q[y | cluster k] = sum_i r_ik * y_i / sum_i r_ik
        responsibilities = self._bgmm.predict_proba(X_scaled)  # (n, K_max)
        self._enrichments = np.zeros(self.k_max, dtype=float)
        for k in range(self.k_max):
            r_k = responsibilities[:, k]
            denom = r_k.sum()
            if denom > 1e-10:
                self._enrichments[k] = float((r_k * y).sum() / denom)

        # Re-use parent's _gmm slot with a stub so score() can work without override
        self._gmm = self._bgmm  # BayesianGMM has predict_proba → compatible

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._bgmm is None:
            raise RuntimeError("Call fit() before score().")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        resp = self._bgmm.predict_proba(X_scaled)
        return resp @ self._enrichments

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def _make_clone(self, seed_offset: int) -> "LPIVariational":
        return LPIVariational(
            k_max=self.k_max,
            scaler=self.scaler,
            random_state=self.random_state + seed_offset,
            weight_concentration_prior=self.weight_concentration_prior,
            covariance_type=self.covariance_type,
            n_bootstrap=self.n_bootstrap,
            n_components_range=self.n_components_range,
        )

    def fit_predict_cv(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        return _cv_with_factory(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            cv,
            factory=lambda fold_idx: self._make_clone(fold_idx + 1),
            class_name="LPIVariational",
            random_state=self.random_state,
        )

    @property
    def n_params_effective(self) -> int:
        if self._bgmm is None or self._effective_k is None:
            return 0
        d = self._bgmm.means_.shape[1]
        return self._effective_k * (d + d * (d + 1) // 2 + 1)


# =============================================================================
# Extension 3 — LPIBayesian
# =============================================================================


class LPIBayesian(LPIDetector):
    """
    LPI with bootstrap uncertainty quantification (Extension 3).

    Motivaton:
        The enrichment f_k = rare-class rate in cluster k is estimated from a
        finite, noisy label set. A single-point estimate ignores this uncertainty.
        In safety-critical systems (satellites), the operator needs to distinguish
        between "LPI=0.8 (tight CI)" and "LPI=0.8 (wide CI — only 3 anomalies
        in cluster)".

    Method:
        1. Fit GMM once (reuses LPIDetector BIC bootstrap for K selection).
        2. Bootstrap the enrichment: resample (X_train, y_train) N times,
           recompute f_k^(b) for each resample b.
        3. score(X) returns mean score over all B enrichment vectors.
        4. score_with_uncertainty(X) returns (mean, std, 90 % CI bounds).

    This captures posterior uncertainty about f_k | finite_labels
    without requiring a full Bayesian GMM (cheaper, more interpretable).
    """

    def __init__(
        self,
        n_components_range: tuple[int, int] = (2, 15),
        n_bootstrap: int = 20,
        scaler: str = "robust",
        random_state: int = 42,
        n_bootstrap_bayes: int = 50,
    ) -> None:
        super().__init__(n_components_range, n_bootstrap, scaler, random_state)
        self.n_bootstrap_bayes = n_bootstrap_bayes
        self._bootstrap_enrichments: Optional[np.ndarray] = None  # (B, K)
        self._mean_enrichments: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LPIBayesian":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # 1. Fit GMM once (scale + BIC + final GMM) using parent's logic
        self._scaler = self._make_scaler()
        X_scaled = self._scaler.fit_transform(X)
        self._best_k, _ = self._select_k_by_bic(X_scaled)
        print(f"[LPIBayesian] BIC K={self._best_k}  B={self.n_bootstrap_bayes}")

        self._gmm = GaussianMixture(
            n_components=self._best_k,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=300,
            n_init=5,
            reg_covar=1e-6,
        )
        self._gmm.fit(X_scaled)
        hard_assignments = self._gmm.predict(X_scaled)

        # 2. Bootstrap enrichments: resample labels, recompute f_k
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        boot_enrichments = np.zeros((self.n_bootstrap_bayes, self._best_k), dtype=float)

        for b in range(self.n_bootstrap_bayes):
            idx = rng.choice(n, size=n, replace=True)
            y_boot = y[idx]
            ha_boot = hard_assignments[idx]
            for k in range(self._best_k):
                mask = ha_boot == k
                if mask.sum() > 0:
                    boot_enrichments[b, k] = float(y_boot[mask].mean())

        self._bootstrap_enrichments = boot_enrichments
        self._mean_enrichments = boot_enrichments.mean(axis=0)
        self._enrichments = self._mean_enrichments  # parent compat

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns mean LPI score averaged over bootstrap enrichments."""
        if self._gmm is None:
            raise RuntimeError("Call fit() before score().")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        resp = self._gmm.predict_proba(X_scaled)  # (n, K)
        return resp @ self._mean_enrichments

    def score_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean_score, std_score, ci_90) per sample.

        ci_90 : (n_samples, 2) array with [5th, 95th] percentile bounds.
        """
        if self._bootstrap_enrichments is None:
            raise RuntimeError("Call fit() before score_with_uncertainty().")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        resp = self._gmm.predict_proba(X_scaled)  # (n, K)
        # all_scores: (n, B)
        all_scores = resp @ self._bootstrap_enrichments.T
        mean_s = all_scores.mean(axis=1)
        std_s = all_scores.std(axis=1)
        ci = np.percentile(all_scores, [5, 95], axis=1).T  # (n, 2)
        return mean_s, std_s, ci

    def _make_clone(self, seed_offset: int) -> "LPIBayesian":
        return LPIBayesian(
            n_components_range=self.n_components_range,
            n_bootstrap=self.n_bootstrap,
            scaler=self.scaler,
            random_state=self.random_state + seed_offset,
            n_bootstrap_bayes=self.n_bootstrap_bayes,
        )

    def fit_predict_cv(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        return _cv_with_factory(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            cv,
            factory=lambda fold_idx: self._make_clone(fold_idx + 1),
            class_name="LPIBayesian",
            random_state=self.random_state,
        )

    @property
    def n_params_effective(self) -> int:
        if self._gmm is None:
            return 0
        d = self._gmm.means_.shape[1]
        gmm_params = self._best_k * (d + d * (d + 1) // 2 + 1)
        # Each bootstrap gives K enrichment values
        return gmm_params + self.n_bootstrap_bayes * self._best_k


# =============================================================================
# Extension 4 — LPIHierarchical
# =============================================================================


class LPIHierarchical(LPIDetector):
    """
    Two-level hierarchical GMM enrichment (Extension 4).

    Architecture:
        Level 1: GMM_macro (K_macro components, BIC-selected in [k_macro_min, k_macro_max])
        Level 2: For each macro-cluster k with ≥ min_cluster_size samples,
                 GMM_micro_k (K_micro components, fixed) is fit on that subset.
        Score:
            LPI_hier(x) = sum_k P_macro(k|x) * LPI_micro_k(x)
            LPI_micro_k(x) = sum_j P_micro_k(j|x) * f_{k,j}

    Advantage over v1:
        Handles situations where the anomaly population is heterogeneous
        (e.g., thermal anomalies cluster differently from power anomalies).
        The two-level structure discovers and exploits this substructure.

    Reference: Jordan & Jacobs (1994), Hierarchical Mixtures of Experts;
               Titterington (1990), Some recent research in the analysis of mixture distributions.
    """

    def __init__(
        self,
        n_components_range: tuple[int, int] = (2, 15),
        n_bootstrap: int = 20,
        scaler: str = "robust",
        random_state: int = 42,
        k_macro_range: tuple[int, int] = (2, 6),
        k_micro: int = 3,
        min_cluster_size: int = 15,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(n_components_range, n_bootstrap, scaler, random_state)
        self.k_macro_range = k_macro_range
        self.k_micro = k_micro
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha

        self._gmm_macro: Optional[GaussianMixture] = None
        self._gmm_micros: dict[int, GaussianMixture] = {}
        self._micro_enrichments: dict[int, np.ndarray] = {}
        self._macro_enrichments: Optional[np.ndarray] = None  # level-1 enrichments

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LPIHierarchical":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # 1. Scale
        self._scaler = self._make_scaler()
        X_scaled = self._scaler.fit_transform(X)

        # 2. BIC K selection for macro GMM
        # Use a restricted range for macro level (fewer, broader clusters)
        _orig_range = self.n_components_range
        self.n_components_range = self.k_macro_range
        self._best_k, _ = self._select_k_by_bic(X_scaled)
        self.n_components_range = _orig_range  # restore
        k_macro = self._best_k
        print(f"[LPIHierarchical] K_macro={k_macro}")

        # 3. Fit macro GMM
        self._gmm_macro = GaussianMixture(
            n_components=k_macro,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=300,
            n_init=5,
            reg_covar=1e-6,
        )
        self._gmm_macro.fit(X_scaled)
        macro_assignments = self._gmm_macro.predict(X_scaled)

        # Macro-level enrichments (v1 formula — Level 1 signal)
        self._macro_enrichments = np.zeros(k_macro, dtype=float)
        for k in range(k_macro):
            mask = macro_assignments == k
            if mask.sum() > 0:
                self._macro_enrichments[k] = float(y[mask].mean())

        # 4. For each macro cluster, fit micro GMM
        self._gmm_micros = {}
        self._micro_enrichments = {}

        for k in range(k_macro):
            mask = macro_assignments == k
            X_k = X_scaled[mask]
            y_k = y[mask]
            n_k = mask.sum()

            if n_k < self.min_cluster_size:
                logger.debug(
                    "Macro cluster %d: only %d samples (<min=%d) — skip micro",
                    k, n_k, self.min_cluster_size,
                )
                # Fallback: single enrichment (no sub-clustering)
                self._micro_enrichments[k] = np.array([float(y_k.mean())])
                continue

            k_micro_eff = min(self.k_micro, max(2, n_k // 10))
            gmm_micro = GaussianMixture(
                n_components=k_micro_eff,
                covariance_type="diag",  # fewer params for small clusters
                random_state=self.random_state + k + 1,
                max_iter=200,
                n_init=3,
                reg_covar=1e-6,
            )
            try:
                gmm_micro.fit(X_k)
            except Exception as exc:
                logger.warning("Micro GMM failed for cluster %d: %s", k, exc)
                self._micro_enrichments[k] = np.array([float(y_k.mean())])
                continue

            self._gmm_micros[k] = gmm_micro
            micro_assignments = gmm_micro.predict(X_k)
            enr = np.zeros(k_micro_eff, dtype=float)
            for j in range(k_micro_eff):
                m_j = micro_assignments == j
                if m_j.sum() > 0:
                    enr[j] = float(y_k[m_j].mean())
            self._micro_enrichments[k] = enr

        # Set parent attributes for interface compatibility
        self._gmm = self._gmm_macro
        self._enrichments = self._macro_enrichments

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._gmm_macro is None:
            raise RuntimeError("Call fit() before score().")
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)

        macro_resp = self._gmm_macro.predict_proba(X_scaled)  # (n, K_macro)
        n_samples = len(X_scaled)
        scores = np.zeros(n_samples, dtype=float)

        for k in range(self._best_k):
            p_macro_k = macro_resp[:, k]  # (n,)
            enr_k = self._micro_enrichments[k]

            if k in self._gmm_micros:
                micro_resp = self._gmm_micros[k].predict_proba(X_scaled)  # (n, K_micro)
                # Truncate to enr_k size (in case k_micro_eff < k_micro)
                micro_resp = micro_resp[:, : len(enr_k)]
                # Renormalise rows (in case of truncation)
                row_sum = micro_resp.sum(axis=1, keepdims=True).clip(1e-10)
                micro_resp = micro_resp / row_sum
                lpi_micro = micro_resp @ enr_k  # (n,)
            else:
                # Single enrichment — no sub-clustering
                lpi_micro = np.full(n_samples, enr_k[0])

            # alpha blending: macro enrichment + micro refinement
            lpi_macro_k = self._macro_enrichments[k]
            scores += p_macro_k * (
                self.alpha * lpi_macro_k + (1 - self.alpha) * lpi_micro
            )

        return scores

    def _make_clone(self, seed_offset: int) -> "LPIHierarchical":
        return LPIHierarchical(
            n_components_range=self.n_components_range,
            n_bootstrap=self.n_bootstrap,
            scaler=self.scaler,
            random_state=self.random_state + seed_offset,
            k_macro_range=self.k_macro_range,
            k_micro=self.k_micro,
            min_cluster_size=self.min_cluster_size,
            alpha=self.alpha,
        )

    def fit_predict_cv(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        return _cv_with_factory(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            cv,
            factory=lambda fold_idx: self._make_clone(fold_idx + 1),
            class_name="LPIHierarchical",
            random_state=self.random_state,
        )

    @property
    def n_params_effective(self) -> int:
        if self._gmm_macro is None:
            return 0
        d = self._gmm_macro.means_.shape[1]
        k_macro = self._best_k
        macro_p = k_macro * (d + d * (d + 1) // 2 + 1)
        micro_p = sum(
            len(enr) * (d + d + 1)  # diag covariance: d mean + d var + weight
            for enr in self._micro_enrichments.values()
        )
        return macro_p + micro_p


# =============================================================================
# Extension 5 — LPIOnline
# =============================================================================


class LPIOnline(LPIDetector):
    """
    Online / streaming LPI with incremental parameter updates (Extension 5).

    Key operational motivation:
        Satellite anomaly patterns drift over mission lifetime (thermal aging,
        component wear, orbital perturbations). Batch models become stale and
        require expensive full retraining. LPIOnline updates parameters with
        each new telemetry window without discarding historical knowledge.

    Algorithm (Online EM with exponential forgetting):
        1. Fit initial GMM on the first batch (or full training set for fit()).
        2. For each subsequent batch:
           a. Re-fit GMM with warm_start=True and reduced max_iter.
           b. Recompute hard cluster assignments for the batch.
           c. Update enrichments with EMA:
                f_k ← (1 - forgetting_rate) * f_k  +  forgetting_rate * f_k_batch
        3. score() uses the current (up-to-date) GMM + enrichments.

    update_batch(X_new, y_new) is the key public API for production deployment.

    Reference: Cappé & Moulines (2009), Online EM Algorithm for Hidden Markov Models.
    """

    def __init__(
        self,
        n_components_range: tuple[int, int] = (2, 15),
        n_bootstrap: int = 20,
        scaler: str = "robust",
        random_state: int = 42,
        batch_size: int = 200,
        forgetting_rate: float = 0.15,
        n_update_iter: int = 20,
    ) -> None:
        super().__init__(n_components_range, n_bootstrap, scaler, random_state)
        self.batch_size = batch_size
        self.forgetting_rate = forgetting_rate
        self.n_update_iter = n_update_iter
        self._n_batches_seen: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LPIOnline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = len(X)

        # 1. Scale (fit on all data — assumes full training set available initially)
        self._scaler = self._make_scaler()
        X_scaled = self._scaler.fit_transform(X)

        # 2. Initial GMM: BIC selection on first batch (or all data if small)
        init_size = min(self.batch_size, n)
        X_init = X_scaled[:init_size]
        self._best_k, _ = self._select_k_by_bic(X_init)
        print(
            f"[LPIOnline] BIC K={self._best_k}  "
            f"batch_size={self.batch_size}  forget={self.forgetting_rate}"
        )

        self._gmm = GaussianMixture(
            n_components=self._best_k,
            covariance_type="full",
            warm_start=False,
            random_state=self.random_state,
            max_iter=300,
            n_init=5,
            reg_covar=1e-6,
        )
        self._gmm.fit(X_init)

        # Initial enrichments from first batch
        ha = self._gmm.predict(X_init)
        self._enrichments = np.zeros(self._best_k, dtype=float)
        for k in range(self._best_k):
            mask = ha == k
            if mask.sum() > 0:
                self._enrichments[k] = float(y[:init_size][mask].mean())
        self._n_batches_seen = 1

        # 3. Simulate streaming: update on remaining batches
        start = init_size
        while start < n:
            end = min(start + self.batch_size, n)
            self.update_batch(X_scaled[start:end], y[start:end], already_scaled=True)
            start = end

        return self

    def update_batch(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        already_scaled: bool = False,
    ) -> "LPIOnline":
        """
        Incrementally update the model with a new data batch.

        Parameters
        ----------
        X_new : (n_new, n_features) feature matrix
        y_new : (n_new,) labels (1 = anomaly)
        already_scaled : bool
            True if X_new is already RobustScaler-transformed (internal use).
        """
        if self._gmm is None:
            raise RuntimeError("Call fit() before update_batch().")

        X_new = np.asarray(X_new, dtype=float)
        y_new = np.asarray(y_new, dtype=int)

        if not already_scaled:
            X_new = self._scaler.transform(X_new)

        # Skip batches too small to fit a GMM (need ≥ 2 * K samples)
        if len(X_new) < max(2 * self._best_k, 4):
            logger.debug("Batch too small (%d samples) for K=%d — skipping", len(X_new), self._best_k)
            return self

        # Re-fit GMM with warm_start for a few iterations
        self._gmm.warm_start = True
        self._gmm.max_iter = self.n_update_iter
        try:
            self._gmm.fit(X_new)
        except Exception as exc:
            logger.warning("Online GMM update failed: %s — skipping batch", exc)
            return self
        finally:
            self._gmm.warm_start = False
            self._gmm.max_iter = 300

        # EMA enrichment update
        ha_new = self._gmm.predict(X_new)
        for k in range(self._best_k):
            mask = ha_new == k
            if mask.sum() > 0:
                f_k_batch = float(y_new[mask].mean())
                self._enrichments[k] = (
                    (1 - self.forgetting_rate) * self._enrichments[k]
                    + self.forgetting_rate * f_k_batch
                )

        self._n_batches_seen += 1
        return self

    def _make_clone(self, seed_offset: int) -> "LPIOnline":
        return LPIOnline(
            n_components_range=self.n_components_range,
            n_bootstrap=self.n_bootstrap,
            scaler=self.scaler,
            random_state=self.random_state + seed_offset,
            batch_size=self.batch_size,
            forgetting_rate=self.forgetting_rate,
            n_update_iter=self.n_update_iter,
        )

    def fit_predict_cv(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        return _cv_with_factory(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int),
            cv,
            factory=lambda fold_idx: self._make_clone(fold_idx + 1),
            class_name="LPIOnline",
            random_state=self.random_state,
        )

    @property
    def n_params_effective(self) -> int:
        if self._gmm is None:
            return 0
        d = self._gmm.means_.shape[1]
        return self._best_k * (d + d * (d + 1) // 2 + 1)
