"""
eval_discrete_fixed.py
======================
Corrected discrete-set evaluation for CNN vs CNN+XGBoost.

Key fixes vs original eval_discrete.py:
  1. Loads saved aux_clf weights (--aux-weights) for true CNN-only comparison.
     Without this both models route through the same XGBoost path.
  2. Score distributions are printed/plotted before ROC to catch saturation.
  3. FAR is computed correctly: N_noise_above_thr / background_duration_s.
  4. Sensitivity uses 50th percentile distance (D50), not D_max.
  5. Debug checks on embedding non-triviality.
"""

import os
import logging
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

import h5py
import torch
import torch.nn as nn
import xgboost as xgb
from tqdm import tqdm

from apply import get_coherent_network, get_coincident_network, get_base_network, dtype


# ══════════════════════════════════════════════════════════════════════════
# 1. Dataset (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class DiscreteTestDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, group_name='testing',
                 snr=None, snr_range=(5., 15.), device='cpu', seed=0):
        super().__init__()
        self.device = device
        self.snr_fixed = snr
        self.snr_range = snr_range
        self.rng = np.random.default_rng(seed)

        with h5py.File(filepath, 'r') as f:
            # ── auto-detect HDF5 layout ────────────────────────────────
            if group_name in f:
                grp = f[group_name]
                noises    = grp['noises'][()]
                waveforms = grp['waveforms'][()]
                logging.info(f"Layout B: {len(waveforms)} signal + {len(noises)} noise "
                             f"from group '{group_name}' "
                             f"(noises={len(noises)}, waveforms={len(waveforms)})")
            else:
                fallback = 'validation'
                if fallback in f:
                    logging.warning(f"Group '{group_name}' not found; using '{fallback}'.")
                    grp = f[fallback]
                    noises    = grp['noises'][()]
                    waveforms = grp['waveforms'][()]
                else:
                    raise KeyError(f"Neither '{group_name}' nor 'validation' found. "
                                   f"Available: {list(f.keys())}")

        self.noises    = torch.from_numpy(noises).to(dtype=dtype, device=device)
        self.waveforms = torch.from_numpy(waveforms).to(dtype=dtype, device=device)
        # Allow more noise than waveforms (Layout B: noises=2N, waveforms=N)
        self.N_sig = len(self.waveforms)
        self.N_noi = len(self.noises)

        if self.snr_fixed is None:
            self.snrs = self.rng.uniform(*self.snr_range, size=self.N_sig).astype(np.float32)
        else:
            self.snrs = np.full(self.N_sig, self.snr_fixed, dtype=np.float32)

    def __len__(self):
        return self.N_sig + self.N_noi

    def __getitem__(self, i):
        if i < self.N_sig:
            snr = float(self.snrs[i])
            x   = self.noises[i] + snr * self.waveforms[i]
            return x, torch.tensor(1, dtype=torch.long)
        else:
            x = self.noises[i - self.N_sig]
            return x, torch.tensor(0, dtype=torch.long)

    @property
    def snr_per_sample(self):
        return np.concatenate([self.snrs, np.zeros(self.N_noi)])


# ══════════════════════════════════════════════════════════════════════════
# 2. Auxiliary classifier (same architecture as train.py)
# ══════════════════════════════════════════════════════════════════════════

def build_aux_classifier(embedding_dim, device):
    return nn.Sequential(
        nn.Linear(embedding_dim, 64),
        nn.Dropout(p=0.3),
        nn.ELU(),
        nn.Linear(64, 2),
        nn.Softmax(dim=1),
    ).to(dtype=dtype, device=device)


# ══════════════════════════════════════════════════════════════════════════
# 3. Score collection  ← KEY FIXES HERE
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_cnn_scores(cnn, aux_clf, dataset, batch_size=256, device='cpu', verbose=False):
    """
    CNN-only scores: CNN trunk → saved aux_clf → P(signal).
    FIX: aux_clf must be loaded from best_aux_clf.pt, not randomly initialised.
    """
    cnn.eval(); aux_clf.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores, all_labels = [], []
    for x, y in tqdm(loader, desc="CNN scores", disable=not verbose, ascii=True):
        x   = x.to(dtype=dtype, device=device)
        emb = cnn(x)
        # Column 0 = P(signal) because wave_label = [1,0]
        prob = aux_clf(emb)[:, 0]
        all_scores.append(prob.cpu().numpy())
        all_labels.append(y.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # ── Debug: score distribution check ───────────────────────────────────
    sig_sc = scores[labels == 1]
    noi_sc = scores[labels == 0]
    logging.info(f"CNN scores — signal: mean={sig_sc.mean():.3f} std={sig_sc.std():.3f}  "
                 f"noise: mean={noi_sc.mean():.3f} std={noi_sc.std():.3f}")
    if sig_sc.std() < 0.01 and noi_sc.std() < 0.01:
        logging.warning("CNN scores near-constant — aux_clf may be random init or saturated!")

    return labels, scores, dataset.snr_per_sample


@torch.no_grad()
def collect_xgb_scores(cnn, xgb_model, dataset, batch_size=256, device='cpu', verbose=False):
    """
    CNN+XGBoost scores: CNN trunk → XGBoost → P(signal).
    """
    cnn.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_labels = [], []
    for x, y in tqdm(loader, desc="XGB scores", disable=not verbose, ascii=True):
        x = x.to(dtype=dtype, device=device)
        all_emb.append(cnn(x).cpu().numpy())
        all_labels.append(y.numpy())
    embeddings = np.vstack(all_emb)
    labels     = np.concatenate(all_labels)

    # ── Debug: embedding health check ─────────────────────────────────────
    logging.info(f"Embeddings — mean={embeddings.mean():.4f}  std={embeddings.std():.4f}")
    dead = (embeddings.std(axis=0) < 1e-4).sum()
    if dead:
        logging.warning(f"  {dead} near-constant embedding dimensions — training may have failed")

    logging.info("Running XGBoost inference…")
    scores = xgb_model.predict_proba(embeddings)[:, 1]

    sig_sc = scores[labels == 1]
    noi_sc = scores[labels == 0]
    logging.info(f"XGB scores — signal: mean={sig_sc.mean():.3f} std={sig_sc.std():.3f}  "
                 f"noise: mean={noi_sc.mean():.3f} std={noi_sc.std():.3f}")
    if abs(sig_sc.mean() - noi_sc.mean()) < 0.05:
        logging.warning("XGB signal/noise means nearly identical — XGBoost may not be learning!")

    return labels, scores, dataset.snr_per_sample


# ══════════════════════════════════════════════════════════════════════════
# 4. FAR and sensitivity helpers  ← CORE FIXES
# ══════════════════════════════════════════════════════════════════════════

def compute_far_curve(noise_scores, background_duration_s, step_size_s=0.1):
    """
    Correct FAR computation for a sliding-window search.

    Parameters
    ----------
    noise_scores : array of scores for all noise (background) windows
    background_duration_s : total duration of background data in seconds
    step_size_s : sliding window step in seconds (default 0.1s)

    Returns
    -------
    thresholds : array of threshold values (descending)
    far_per_second : FAR in events/second at each threshold
    far_per_month : FAR in events/month at each threshold

    Explanation of the fix
    ----------------------
    Original: FAR = N_triggers / T_seconds  (counts clustered triggers, not windows)
    Problem:  clustering introduces threshold-dependent event merging, biasing FAR.

    Correct:  FAR = N_noise_windows_above_thr / T_seconds
    This is the expected number of background windows per second exceeding the
    threshold — directly comparable to a foreground detection rate.

    NOTE: if you have clustered triggers rather than per-window scores, use:
        FAR_clustered = N_clusters_above_thr / T_seconds
    but ensure T_seconds is background-only time, not total time.
    """
    SECONDS_PER_MONTH = 30 * 24 * 3600

    thresholds = np.unique(noise_scores)[::-1]  # descending

    far_per_s = np.array([
        (noise_scores >= thr).sum() / background_duration_s
        for thr in thresholds
    ])
    far_per_month = far_per_s * SECONDS_PER_MONTH

    return thresholds, far_per_s, far_per_month


def compute_sensitivity_d50(signal_scores, signal_distances, noise_scores,
                            background_duration_s,
                            far_targets_per_month=(0.01, 0.1, 1.0),
                            percentile=50):
    """
    Compute sensitive distance D_p at a given FAR target.

    FIX 1: uses percentile distance (D50 = median), not D_max.
    FIX 2: FAR computed from noise window scores, not event count.

    Parameters
    ----------
    signal_scores : P(signal) for each signal sample
    signal_distances : source distance in Mpc for each signal sample
                       (if not available, use np.ones_like and interpret as
                        detection efficiency instead)
    noise_scores : P(signal) for each noise sample
    background_duration_s : duration of background data in seconds
    far_targets_per_month : list of FAR values (events/month) to evaluate at
    percentile : which distance percentile to report (default 50 = median)

    Returns
    -------
    results : dict  {far_target: {'threshold': float, 'distance': float,
                                   'efficiency': float, 'n_found': int}}
    """
    SECONDS_PER_MONTH = 30 * 24 * 3600

    results = {}
    for far_target in far_targets_per_month:
        # Convert FAR target to threshold
        target_per_s = far_target / SECONDS_PER_MONTH
        # Threshold = lowest score such that FAR ≤ target
        # Sort noise scores descending; find cutoff
        sorted_noise = np.sort(noise_scores)[::-1]
        n_noise = len(sorted_noise)

        # FAR(thr) = #{noise_scores >= thr} / T
        # We want the smallest thr where FAR ≤ target_per_s
        n_allowed = int(np.floor(target_per_s * background_duration_s))
        if n_allowed < 1:
            # target FAR too low for this background duration
            threshold = np.inf
        elif n_allowed >= n_noise:
            threshold = 0.0
        else:
            threshold = sorted_noise[n_allowed]   # n_allowed-th highest noise score

        # Detection efficiency at this threshold
        found_mask = signal_scores >= threshold
        n_found = found_mask.sum()
        efficiency = n_found / len(signal_scores)

        # ── D_p: the p-th percentile distance among found signals ─────────
        if n_found == 0:
            d_p = 0.0
        else:
            found_distances = signal_distances[found_mask]
            d_p = np.percentile(found_distances, percentile)

        results[far_target] = dict(
            threshold=float(threshold),
            distance=float(d_p),
            efficiency=float(efficiency),
            n_found=int(n_found),
        )

    return results


# ══════════════════════════════════════════════════════════════════════════
# 5. Plot helpers
# ══════════════════════════════════════════════════════════════════════════

STYLE = {
    'cnn': dict(color='steelblue',  lw=2, ls='-',  label_prefix='CNN-only'),
    'xgb': dict(color='darkorange', lw=2, ls='--', label_prefix='CNN+XGB'),
}


def plot_debug_scores(results_dict, save_path):
    """
    DIAGNOSTIC: histogram of raw scores for signal and noise.
    If CNN-only and CNN+XGB look identical here, the aux_clf is broken.
    """
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for col, (key, (y_true, scores)) in enumerate(results_dict.items()):
        ax = axes[0][col]
        bins = np.linspace(0, 1, 80)
        ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, color='royalblue',
                label=f'Noise (n={int((y_true==0).sum()):,})', density=True)
        ax.hist(scores[y_true == 1], bins=bins, alpha=0.6, color='crimson',
                label=f'Signal (n={int((y_true==1).sum()):,})', density=True)
        ax.set_yscale('log')
        ax.set(xlabel='Score P(signal)', ylabel='Density (log)',
               title=f'{STYLE[key]["label_prefix"]}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Print useful stats
        sig = scores[y_true == 1]; noi = scores[y_true == 0]
        ax.set_title(
            f'{STYLE[key]["label_prefix"]}\n'
            f'signal μ={sig.mean():.3f} σ={sig.std():.3f}   '
            f'noise μ={noi.mean():.3f} σ={noi.std():.3f}',
            fontsize=9
        )

    fig.suptitle('Score distributions — debug', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  Debug score distributions → {save_path}")


def plot_roc(results_dict, save_path, title='ROC – Discrete Test Set', show=False):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.4, label='Random')
    summary = {}
    for key, (y_true, scores) in results_dict.items():
        fpr, tpr, thr = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        opt = np.argmax(tpr - fpr)
        s = STYLE[key]
        ax.plot(fpr, tpr, color=s['color'], lw=s['lw'], linestyle=s['ls'],
                label=f"{s['label_prefix']}  AUC={roc_auc:.4f}")
        ax.plot(fpr[opt], tpr[opt], 'o', color=s['color'], ms=8)
        summary[key] = dict(auc=roc_auc, opt_threshold=float(thr[opt]),
                            opt_tpr=float(tpr[opt]), opt_fpr=float(fpr[opt]))

    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel='False Positive Rate', ylabel='True Positive Rate', title=title)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  ROC → {save_path}")
    for k, v in summary.items():
        print(f"   [{k}]  AUC={v['auc']:.4f}   opt_thr={v['opt_threshold']:.4f}  "
              f"TPR={v['opt_tpr']:.3f}  FPR={v['opt_fpr']:.3f}")
    return summary


def plot_det(results_dict, save_path, title='DET – Discrete Test Set', show=False):
    eps = 1e-7
    tick_probs  = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
    tick_locs   = norm.ppf(tick_probs)
    tick_labels = ['0.1%','1%','5%','10%','20%','50%','80%','90%','95%','99%']
    fig, ax = plt.subplots(figsize=(8, 7))
    for key, (y_true, scores) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        fnr = np.clip(1. - tpr, eps, 1 - eps)
        fpr_c = np.clip(fpr, eps, 1 - eps)
        s = STYLE[key]
        ax.plot(norm.ppf(fpr_c), norm.ppf(fnr),
                color=s['color'], lw=s['lw'], linestyle=s['ls'], label=s['label_prefix'])
    ax.set_xticks(tick_locs); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_locs); ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set(xlabel='False Alarm Rate (probit)', ylabel='Miss Rate (probit)', title=title)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  DET → {save_path}")


def plot_snr_binned_roc(results_dict, snr_array, snr_bins, save_path,
                        title='SNR-Binned ROC', show=False):
    n_bins = len(snr_bins)
    n_cols = min(3, n_bins)
    n_rows = (n_bins + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    noise_mask = (snr_array == 0)
    for idx, (lo, hi) in enumerate(snr_bins):
        ax = axes[idx // n_cols][idx % n_cols]
        sig_mask  = (snr_array >= lo) & (snr_array < hi)
        combo     = sig_mask | noise_mask
        n_sig     = sig_mask.sum()
        if n_sig < 5:
            ax.text(0.5, 0.5, f'SNR [{lo},{hi})\n{n_sig} signals',
                    ha='center', va='center', transform=ax.transAxes)
            continue
        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
        for key, (y_true, scores) in results_dict.items():
            fpr, tpr, _ = roc_curve(y_true[combo], scores[combo])
            s = STYLE[key]
            ax.plot(fpr, tpr, color=s['color'], lw=s['lw'], linestyle=s['ls'],
                    label=f"{s['label_prefix']} (AUC={auc(fpr,tpr):.3f})")
        ax.set(xlim=[0,1], ylim=[0,1.02], xlabel='FPR', ylabel='TPR',
               title=f'SNR ∈ [{lo}, {hi})  — {n_sig} signals')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    for ax in axes.flatten()[n_bins:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  SNR-binned ROC → {save_path}")


def plot_far_sensitivity(results_dict, background_duration_s,
                         signal_distances, save_path,
                         far_range_per_month=(1e-3, 1e3),
                         percentile=50, show=False):
    """
    FIX: sensitivity plot using correct FAR and D50.

    Replaces the old sensitivity.py / gen-pre.py workflow.
    """
    SECONDS_PER_MONTH = 30 * 24 * 3600
    far_min, far_max = far_range_per_month

    fig, ax = plt.subplots(figsize=(9, 6))

    for key, (y_true, scores) in results_dict.items():
        noise_scores  = scores[y_true == 0]
        signal_scores = scores[y_true == 1]
        sig_dist      = signal_distances[y_true == 1] if signal_distances is not None \
                        else np.ones(signal_scores.shape)

        # ── compute FAR and D50 across threshold sweep ────────────────────
        sorted_noise = np.sort(noise_scores)
        n_noise = len(sorted_noise)
        far_values, d50_values = [], []

        # Sweep thresholds from easy (low, many detections) to hard (high, few)
        for thr in np.percentile(scores, np.linspace(1, 99, 200)):
            far_ps = (noise_scores >= thr).sum() / background_duration_s
            far_pm = far_ps * SECONDS_PER_MONTH
            if not (far_min <= far_pm <= far_max):
                continue
            found = signal_scores >= thr
            if found.sum() == 0:
                continue
            d50 = np.percentile(sig_dist[found], percentile)
            far_values.append(far_pm)
            d50_values.append(d50)

        if len(far_values) == 0:
            print(f"  [{key}] No points in FAR range — check background duration or threshold.")
            continue

        far_arr = np.array(far_values)
        d50_arr = np.array(d50_values)
        order   = np.argsort(far_arr)
        s = STYLE[key]
        ax.semilogx(far_arr[order], d50_arr[order],
                    color=s['color'], lw=s['lw'], linestyle=s['ls'],
                    label=s['label_prefix'])

    ax.set(xlabel=f'FAR (events / month)',
           ylabel=f'D{percentile} sensitive distance (Mpc)',
           title=f'Sensitivity curve — D{percentile} vs FAR')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, which='both')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  FAR-sensitivity → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Fixed discrete evaluation: CNN vs CNN+XGBoost.'
    )
    # data
    parser.add_argument('testfile')
    parser.add_argument('--group', default='testing')
    # model weights
    parser.add_argument('--cnn-weights', required=True)
    parser.add_argument('--xgb-weights', required=True)
    # ── FIX: new argument for saved aux_clf weights ────────────────────────
    parser.add_argument('--aux-weights', default=None,
                        help='Path to best_aux_clf.pt (saved by train_fixed.py). '
                             'Without this CNN-only results are meaningless.')
    parser.add_argument('--embedding-dim', type=int, default=128)
    # network
    parser.add_argument('--coincident', action='store_true')
    parser.add_argument('--device', default='cpu')
    # injection / SNR
    parser.add_argument('--snr', type=float, default=None)
    parser.add_argument('--snr-range', type=float, nargs=2, default=[5., 15.], metavar=('MIN','MAX'))
    parser.add_argument('--snr-bins', default='5-8,8-10,10-12,12-15,15-20')
    parser.add_argument('--seed', type=int, default=42)
    # sensitivity
    parser.add_argument('--background-duration', type=float, default=None,
                        help='Background data duration in seconds. '
                             'Default: inferred from dataset size × step-size.')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Sliding window step in seconds (default 0.1).')
    parser.add_argument('--sensitivity-percentile', type=int, default=50,
                        help='Distance percentile for sensitivity curve (default 50).')
    # output
    parser.add_argument('--output-dir', default='eval_discrete_fixed')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("FIXED DISCRETE EVALUATION  —  CNN  vs  CNN+XGBoost")
    print("=" * 70)

    # ── resolve group ──────────────────────────────────────────────────────
    group = args.group
    with h5py.File(args.testfile, 'r') as f:
        if group not in f:
            fallback = 'validation'
            if fallback in f:
                logging.warning(f"Group '{group}' not found; using '{fallback}'.")
                group = fallback
            else:
                raise KeyError(f"Neither '{group}' nor 'validation' found.")

    # ── load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading test data: '{args.testfile}' (group: '{group}') …")
    dataset = DiscreteTestDataset(
        filepath=args.testfile, group_name=group,
        snr=args.snr, snr_range=tuple(args.snr_range),
        device=args.device, seed=args.seed,
    )
    print(f"  Signal  : {dataset.N_sig:,}")
    print(f"  Noise   : {dataset.N_noi:,}")
    snr_all = dataset.snr_per_sample

    # ── SNR bins ───────────────────────────────────────────────────────────
    snr_bins = []
    for seg in args.snr_bins.split(','):
        lo, hi = map(float, seg.strip().split('-'))
        snr_bins.append((lo, hi))

    # ── load CNN trunk ─────────────────────────────────────────────────────
    print("\nLoading CNN …")
    num_det = dataset.noises.shape[1]

    if args.coincident:
        wrapped = get_coincident_network(
            path=args.cnn_weights, device=args.device,
            detectors=num_det, regularize=False, embedding_dim=args.embedding_dim)
    else:
        wrapped = get_coherent_network(
            path=args.cnn_weights, device=args.device,
            detectors=num_det, regularize=False, embedding_dim=args.embedding_dim)

    # Unwrap to get raw CNN trunk
    if hasattr(wrapped, 'cnn_network'):
        cnn = wrapped.cnn_network
        logging.info("Extracted CNN trunk via .cnn_network")
    elif hasattr(wrapped, 'cnn'):
        cnn = wrapped.cnn
    elif hasattr(wrapped, 'feature_extractor'):
        cnn = wrapped.feature_extractor
    else:
        cnn = wrapped
    cnn.to(dtype=dtype, device=args.device).eval()
    print("  CNN ✓")

    # ── load / build aux classifier ────────────────────────────────────────
    aux_clf = build_aux_classifier(args.embedding_dim, args.device)
    if args.aux_weights is not None and os.path.isfile(args.aux_weights):
        aux_clf.load_state_dict(torch.load(args.aux_weights, map_location=args.device))
        print(f"  Aux classifier ✓  (loaded from {args.aux_weights})")
    else:
        logging.warning(
            "aux_weights not found — CNN-only uses RANDOM INIT. "
            "CNN-only AUC will be ~0.5. Re-train with train_fixed.py and "
            "pass --aux-weights path/to/best_aux_clf.pt for valid comparison.")
        print("  Aux classifier : random init  (CNN-only results are a sanity check)")
        print("  Tip: save aux_clf weights in train.py and pass --aux-weights")

    # ── load XGBoost ───────────────────────────────────────────────────────
    print("Loading XGBoost …")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_weights)
    print("  XGBoost ✓")

    # ── collect scores ─────────────────────────────────────────────────────
    print("\nRunning inference …")
    y_true_cnn, scores_cnn, _ = collect_cnn_scores(
        cnn, aux_clf, dataset,
        batch_size=args.batch_size, device=args.device, verbose=args.verbose)
    y_true_xgb, scores_xgb, _ = collect_xgb_scores(
        cnn, xgb_model, dataset,
        batch_size=args.batch_size, device=args.device, verbose=args.verbose)

    assert np.array_equal(y_true_cnn, y_true_xgb), "Label mismatch — this should never happen."
    y_true = y_true_cnn

    print(f"\n  Total  : {len(y_true):,}")
    print(f"  Signal : {int(y_true.sum()):,}")
    print(f"  Noise  : {int((1-y_true).sum()):,}")

    # Quick score sanity summary
    for name, sc in [("CNN", scores_cnn), ("XGB", scores_xgb)]:
        sig, noi = sc[y_true==1], sc[y_true==0]
        print(f"  {name:>3} scores — "
              f"signal: {sig.mean():.3f}±{sig.std():.3f}  "
              f"noise: {noi.mean():.3f}±{noi.std():.3f}")

    results = {
        'cnn': (y_true, scores_cnn),
        'xgb': (y_true, scores_xgb),
    }

    # ── background duration for FAR ────────────────────────────────────────
    # If not supplied, estimate from number of noise samples × step size.
    # This is a lower bound — use actual background file duration if available.
    if args.background_duration is not None:
        bg_duration_s = args.background_duration
    else:
        bg_duration_s = dataset.N_noi * args.step_size
        logging.warning(f"background_duration not set — estimating as "
                        f"{dataset.N_noi} × {args.step_size}s = {bg_duration_s}s. "
                        f"Pass --background-duration for accuracy.")

    print(f"\n  Background duration: {bg_duration_s:.1f}s  ({bg_duration_s/86400:.3f} days)")

    # Fake signal distances (uniform random) if none available.
    # Replace with real distance array from the injection file if possible.
    signal_distances = np.random.uniform(100, 500, size=int(y_true.sum()))
    logging.warning("Signal distances not available — using synthetic uniform [100,500] Mpc. "
                    "Pass real distances for a meaningful sensitivity curve.")

    # ── plots ──────────────────────────────────────────────────────────────
    out = args.output_dir
    print("\n" + "─" * 60)
    print("Generating plots …")
    print("─" * 60)

    plot_debug_scores(
        results,
        save_path=os.path.join(out, 'debug_score_distribution.png'))

    roc_summary = plot_roc(
        results,
        save_path=os.path.join(out, 'roc_discrete.png'))

    plot_det(
        results,
        save_path=os.path.join(out, 'det_discrete.png'))

    plot_snr_binned_roc(
        results, snr_all, snr_bins,
        save_path=os.path.join(out, 'snr_binned_discrete.png'))

    plot_far_sensitivity(
        results,
        background_duration_s=bg_duration_s,
        signal_distances=np.concatenate([signal_distances, np.zeros(dataset.N_noi)]),
        save_path=os.path.join(out, 'far_sensitivity.png'),
        percentile=args.sensitivity_percentile)

    # ── FAR summary at standard targets ───────────────────────────────────
    print("\n" + "─" * 60)
    print("FAR / Sensitivity summary:")
    print("─" * 60)
    for key, (y_true_k, scores_k) in results.items():
        res = compute_sensitivity_d50(
            signal_scores=scores_k[y_true_k == 1],
            signal_distances=signal_distances,
            noise_scores=scores_k[y_true_k == 0],
            background_duration_s=bg_duration_s,
            far_targets_per_month=[0.01, 0.1, 1.0],
            percentile=args.sensitivity_percentile,
        )
        print(f"\n  [{STYLE[key]['label_prefix']}]")
        for far_t, v in res.items():
            print(f"    FAR={far_t}/month  thr={v['threshold']:.4f}  "
                  f"efficiency={v['efficiency']:.3f}  "
                  f"D{args.sensitivity_percentile}={v['distance']:.1f} Mpc  "
                  f"n_found={v['n_found']:,}")

    print("\n" + "=" * 70)
    print("✓  Evaluation complete!")
    print(f"   Output: {os.path.abspath(out)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
