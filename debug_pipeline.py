"""
debug_pipeline.py
=================
Standalone diagnostic script.  Run this BEFORE retraining to understand
exactly which of the five issues apply to your current checkpoints.

Usage
-----
python debug_pipeline.py \\
    --dataset      /path/to/test.h5 \\
    --cnn-weights  best_cnn.pt \\
    --xgb-weights  best_xgboost.json \\
    --aux-weights  best_aux_clf.pt \\       # optional
    --embedding-dim 256 \\
    --output-dir   ./debug_output \\
    --device       cuda
"""

import os
import argparse
import logging

import numpy as np
import h5py
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

dtype = torch.float32


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def load_dataset(fpath, group_name='validation', snr_range=(5., 15.), device='cpu'):
    """Load a discrete dataset and return signal+noise tensors and SNRs."""
    with h5py.File(fpath, 'r') as f:
        if group_name not in f:
            group_name = 'validation' if 'validation' in f else list(f.keys())[0]
        grp = f[group_name]
        noises    = grp['noises'][()]
        waveforms = grp['waveforms'][()]

    noises    = torch.from_numpy(noises).to(dtype=dtype, device=device)
    waveforms = torch.from_numpy(waveforms).to(dtype=dtype, device=device)

    rng = np.random.default_rng(0)
    N = len(waveforms)
    snrs = rng.uniform(*snr_range, size=N).astype(np.float32)

    # Build samples
    signal_x = noises[:N] + torch.from_numpy(snrs).to(dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1) * waveforms
    noise_x  = noises

    samples = torch.cat([signal_x, noise_x], dim=0)
    labels  = np.concatenate([np.ones(N), np.zeros(len(noises))])

    return samples, labels, np.concatenate([snrs, np.zeros(len(noises))])


def build_aux_classifier(embedding_dim, device):
    return nn.Sequential(
        nn.Linear(embedding_dim, 64),
        nn.Dropout(p=0.3),
        nn.ELU(),
        nn.Linear(64, 2),
        nn.Softmax(dim=1),
    ).to(dtype=dtype, device=device)


@torch.no_grad()
def extract_embeddings(cnn, samples, batch_size=256, device='cpu'):
    cnn.eval()
    embs = []
    for i in range(0, len(samples), batch_size):
        x = samples[i:i+batch_size].to(dtype=dtype, device=device)
        embs.append(cnn(x).cpu().numpy())
    return np.vstack(embs)


@torch.no_grad()
def get_cnn_scores(cnn, aux_clf, samples, batch_size=256, device='cpu'):
    cnn.eval(); aux_clf.eval()
    scores = []
    for i in range(0, len(samples), batch_size):
        x = samples[i:i+batch_size].to(dtype=dtype, device=device)
        s = aux_clf(cnn(x))[:, 0].cpu().numpy()
        scores.append(s)
    return np.concatenate(scores)


# ──────────────────────────────────────────────────────────────────────────
# Diagnostic functions
# ──────────────────────────────────────────────────────────────────────────

def check_1_embedding_health(embeddings, labels, out_dir):
    """Issue 3+1: Are the embeddings non-trivial and separable?"""
    print("\n" + "=" * 60)
    print("CHECK 1: Embedding health")
    print("=" * 60)

    std_per_dim = embeddings.std(axis=0)
    dead = (std_per_dim < 1e-4).sum()
    print(f"  Embedding dim      : {embeddings.shape[1]}")
    print(f"  Global mean        : {embeddings.mean():.4f}")
    print(f"  Global std         : {embeddings.std():.4f}")
    print(f"  Dead dimensions    : {dead} / {embeddings.shape[1]}")

    if dead > embeddings.shape[1] * 0.5:
        print("  ⚠  > 50% dead dimensions — CNN training likely failed or embeddings are collapsed")
    elif dead == 0:
        print("  ✓  No dead dimensions")

    sig_emb = embeddings[labels > 0.5]
    noi_emb = embeddings[labels < 0.5]
    separation = np.linalg.norm(sig_emb.mean(0) - noi_emb.mean(0)) / (embeddings.std() + 1e-8)
    print(f"  Signal/noise centroid separation: {separation:.3f}")
    if separation < 0.5:
        print("  ⚠  Low separation — embeddings don't distinguish signal from noise")
    elif separation > 2.0:
        print("  ✓  Good separation")
    else:
        print("  ○  Moderate separation — XGBoost may still find useful structure")

    # Logistic regression sanity check (should approach XGBoost performance)
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(embeddings, labels, test_size=0.2, random_state=0)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_tr, y_tr.astype(int))
    lr_auc = roc_auc_score(y_va, lr.predict_proba(X_va)[:, 1])
    print(f"  Logistic regression AUC on embeddings: {lr_auc:.4f}")
    if lr_auc < 0.6:
        print("  ⚠  LR AUC < 0.6 — embeddings carry little discriminative information")
    elif lr_auc > 0.85:
        print("  ✓  Embeddings are highly discriminative")
    else:
        print("  ○  Moderate LR AUC — XGBoost non-linear kernel may improve this")

    # PCA visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pca = PCA(n_components=2)
    Z = pca.fit_transform(embeddings)
    for lbl, name, col in [(1, 'Signal', 'crimson'), (0, 'Noise', 'royalblue')]:
        mask = labels == lbl
        axes[0].scatter(Z[mask, 0], Z[mask, 1], c=col, alpha=0.3, s=4, label=name)
    axes[0].set(title='PCA of embeddings (first 2 components)',
                xlabel='PC1', ylabel='PC2')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    axes[1].bar(range(min(20, embeddings.shape[1])),
                std_per_dim[:20], color='steelblue', alpha=0.8)
    axes[1].set(title='Std per embedding dimension (first 20)',
                xlabel='Dimension index', ylabel='Std dev')
    axes[1].axhline(1e-4, color='red', ls='--', lw=1, label='dead threshold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(out_dir, 'debug_embeddings.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {path}")

    return lr_auc


def check_2_cnn_scores(cnn_scores, labels, out_dir, name='CNN-only'):
    """Issue 1: Are the CNN-only scores meaningful?"""
    print(f"\n{'='*60}")
    print(f"CHECK 2: {name} score distribution")
    print(f"{'='*60}")

    sig = cnn_scores[labels > 0.5]
    noi = cnn_scores[labels < 0.5]

    print(f"  Score range: [{cnn_scores.min():.4f}, {cnn_scores.max():.4f}]")
    print(f"  Signal  — mean={sig.mean():.4f}  std={sig.std():.4f}  "
          f"median={np.median(sig):.4f}")
    print(f"  Noise   — mean={noi.mean():.4f}  std={noi.std():.4f}  "
          f"median={np.median(noi):.4f}")

    if cnn_scores.std() < 0.005:
        print(f"  ⚠  SCORE VARIANCE NEAR ZERO — {name} output is constant!")
        print("     Diagnosis: aux_clf is randomly initialised OR model output saturated")
        print("     Fix: (a) save best_aux_clf.pt in train.py, (b) load with --aux-weights")
    else:
        auc = roc_auc_score(labels, cnn_scores)
        print(f"  AUC (from scores): {auc:.4f}")
        if auc < 0.55:
            print("  ⚠  AUC near 0.5 — scores are essentially random")
        elif auc > 0.9:
            print("  ✓  High AUC — scores discriminate well")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 80)
    ax.hist(noi, bins=bins, alpha=0.6, color='royalblue', density=True,
            label=f'Noise (n={len(noi):,})')
    ax.hist(sig, bins=bins, alpha=0.6, color='crimson', density=True,
            label=f'Signal (n={len(sig):,})')
    ax.set_yscale('log')
    ax.set(xlabel='Score', ylabel='Density (log)', title=f'{name} score distribution')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, f'debug_scores_{name.lower().replace("+","_").replace(" ","_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {path}")


def check_3_xgb_vs_cnn_roc(cnn_scores, xgb_scores, labels, out_dir):
    """Issue 1 (part 2): Are the two ROC curves actually different?"""
    print(f"\n{'='*60}")
    print("CHECK 3: CNN vs XGB ROC comparison")
    print(f"{'='*60}")

    from sklearn.metrics import roc_curve, auc as roc_auc

    cnn_auc = roc_auc_score(labels, cnn_scores) if cnn_scores.std() > 0.001 else 0.5
    xgb_auc = roc_auc_score(labels, xgb_scores) if xgb_scores.std() > 0.001 else 0.5
    print(f"  CNN-only AUC : {cnn_auc:.4f}")
    print(f"  CNN+XGB  AUC : {xgb_auc:.4f}")
    diff = abs(cnn_auc - xgb_auc)
    print(f"  |ΔAUC|       : {diff:.4f}")

    if diff < 0.001:
        print("  ⚠  IDENTICAL ROC curves — this is the symptom you observed.")
        print("     Most likely cause: CNN-only is using random aux_clf weights,")
        print("     so both 'CNN' and 'XGB' are actually the XGBoost output.")
        print("     Fix: retrain with train_fixed.py (saves best_aux_clf.pt)")
    elif xgb_auc > cnn_auc + 0.01:
        print("  ✓  XGBoost improves over CNN-only — hybrid is working")
    elif cnn_auc > xgb_auc + 0.01:
        print("  ○  CNN-only outperforms XGBoost — consider more XGB training or tuning")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
    for name, sc, col, ls in [
        ('CNN-only', cnn_scores, 'steelblue', '-'),
        ('CNN+XGB',  xgb_scores, 'darkorange', '--'),
    ]:
        if sc.std() > 0.001:
            fpr, tpr, _ = roc_curve(labels, sc)
            ra = roc_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, lw=2, ls=ls, label=f'{name} AUC={ra:.4f}')
        else:
            ax.plot([0,1],[0.5,0.5],color=col,lw=1,ls=':',label=f'{name} (constant output)')
    ax.set(xlim=[0,1],ylim=[0,1.02],xlabel='FPR',ylabel='TPR',title='ROC comparison')
    ax.legend(fontsize=10); ax.grid(True,alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, 'debug_roc_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {path}")


def check_4_far_computation(noise_scores, bg_duration_s, step_size_s=0.1,
                             out_dir='.'):
    """Issues 2+5: FAR computation correctness check."""
    print(f"\n{'='*60}")
    print("CHECK 4: FAR computation")
    print(f"{'='*60}")

    SECONDS_PER_MONTH = 30 * 24 * 3600

    print(f"  Background duration: {bg_duration_s:.0f}s "
          f"({bg_duration_s/86400:.2f} days)")
    print(f"  Number of noise windows: {len(noise_scores):,}")
    print(f"  Windows per second (1/step_size): {1/step_size_s:.1f}")
    print(f"  Effective background coverage: "
          f"{len(noise_scores) * step_size_s:.0f}s "
          f"(should be ≈ bg_duration_s)")

    if abs(len(noise_scores) * step_size_s - bg_duration_s) > 0.3 * bg_duration_s:
        print("  ⚠  Window count × step_size differs from bg_duration by > 30%")
        print("     Check that bg_duration_s matches the actual file you searched")

    print("\n  FAR at representative thresholds (correct formula):")
    print("  FAR = #{noise_scores >= thr} / bg_duration_s * SECONDS_PER_MONTH")
    for thr in [0.3, 0.5, 0.7, 0.9]:
        n_above = (noise_scores >= thr).sum()
        far = n_above / bg_duration_s * SECONDS_PER_MONTH
        print(f"    thr={thr:.1f}  n_above={n_above:7,}  FAR={far:.2f}/month")


def check_5_score_collapse(scores, label='model', out_dir='.'):
    """Issue 5: Detect score collapse (all windows near same value)."""
    print(f"\n{'='*60}")
    print(f"CHECK 5: Score collapse ({label})")
    print(f"{'='*60}")

    print(f"  min={scores.min():.4f}  max={scores.max():.4f}  "
          f"std={scores.std():.4f}  IQR={np.percentile(scores,75)-np.percentile(scores,25):.4f}")

    iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
    if iqr < 0.01:
        print("  ⚠  IQR < 0.01 — SCORE COLLAPSE DETECTED")
        print("     Likely cause: whitening mismatch between train and eval data")
        print("     The CNN/XGBoost outputs a near-constant value for all inputs")
        print("     Fix: ensure training data whitening exactly matches apply.py whitening")
        print("     Check: are waveforms in test.h5 normalised identically to gen.py output?")
    else:
        print(f"  ✓  Score range is non-trivial (IQR={iqr:.4f})")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GW pipeline diagnostic tool.')
    parser.add_argument('--dataset',       required=True)
    parser.add_argument('--cnn-weights',   required=True)
    parser.add_argument('--xgb-weights',   required=True)
    parser.add_argument('--aux-weights',   default=None)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--group',         default='validation')
    parser.add_argument('--snr-range',     type=float, nargs=2, default=[5., 15.])
    parser.add_argument('--device',        default='cpu')
    parser.add_argument('--batch-size',    type=int, default=256)
    parser.add_argument('--bg-duration',   type=float, default=None,
                        help='Background duration in seconds (for FAR check). '
                             'Defaults to n_noise × 0.1s if not set.')
    parser.add_argument('--output-dir',    default='./debug_output')
    parser.add_argument('--verbose',       action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("GW PIPELINE DIAGNOSTIC  — 5 checks")
    print("=" * 70)

    # ── Load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    samples, labels, snrs = load_dataset(
        args.dataset, group_name=args.group,
        snr_range=tuple(args.snr_range), device='cpu',   # keep on cpu for load
    )
    print(f"  Samples: {len(samples):,}  Signal: {int(labels.sum()):,}  Noise: {int((1-labels).sum()):,}")

    # ── Load CNN ───────────────────────────────────────────────────────────
    print(f"\nLoading CNN from {args.cnn_weights}")
    from apply import get_base_network
    num_det = samples.shape[1]
    cnn = get_base_network(path=args.cnn_weights, device=args.device,
                            detectors=num_det, embedding_dim=args.embedding_dim)
    samples = samples.to(device=args.device)

    # ── Extract embeddings ─────────────────────────────────────────────────
    print("Extracting embeddings…")
    embeddings = extract_embeddings(cnn, samples, batch_size=args.batch_size, device=args.device)

    # ── Load aux_clf ───────────────────────────────────────────────────────
    aux_clf = build_aux_classifier(args.embedding_dim, args.device)
    if args.aux_weights and os.path.isfile(args.aux_weights):
        aux_clf.load_state_dict(torch.load(args.aux_weights, map_location=args.device))
        print(f"  Aux classifier loaded from {args.aux_weights}")
    else:
        print("  ⚠  aux_weights not found — CNN-only will use random init (expected AUC≈0.5)")

    # ── Load XGBoost ───────────────────────────────────────────────────────
    print(f"Loading XGBoost from {args.xgb_weights}")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_weights)

    # ── Get scores ─────────────────────────────────────────────────────────
    print("Computing CNN-only scores…")
    cnn_scores = get_cnn_scores(cnn, aux_clf, samples, args.batch_size, args.device)
    print("Computing CNN+XGB scores…")
    xgb_scores = xgb_model.predict_proba(embeddings)[:, 1]

    # ── Run all checks ─────────────────────────────────────────────────────
    check_1_embedding_health(embeddings, labels, args.output_dir)
    check_2_cnn_scores(cnn_scores, labels, args.output_dir, name='CNN-only')
    check_2_cnn_scores(xgb_scores, labels, args.output_dir, name='CNN+XGB')
    check_3_xgb_vs_cnn_roc(cnn_scores, xgb_scores, labels, args.output_dir)

    bg_dur = args.bg_duration if args.bg_duration else int((labels == 0).sum()) * 0.1
    check_4_far_computation(xgb_scores[labels == 0], bg_dur, out_dir=args.output_dir)
    check_5_score_collapse(xgb_scores, label='CNN+XGB', out_dir=args.output_dir)
    check_5_score_collapse(cnn_scores, label='CNN-only', out_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("✓  Diagnostics complete!")
    print(f"   Plots saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
