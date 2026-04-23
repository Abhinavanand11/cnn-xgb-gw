"""
sensitivity_fixed.py
====================
Correct FAR + D50 sensitivity computation for the continuous GW search.

Usage
-----
python sensitivity_fixed.py \\
    --foreground   foreground_preds.hdf \\
    --background   background_preds.hdf \\
    --bg-duration  86400 \\
    --step-size    0.1 \\
    --output-plot  sensitivity_fixed.png \\
    --verbose

Fixes vs original sensitivity.py
---------------------------------
1. FAR = N_noise_windows_above_thr / T_background_seconds
   (not N_triggers / T, which double-counts per-cluster and mis-normalises)

2. D50 = 50th-percentile distance of found signals
   (not D_max = max distance, which is both noisy and optimistic)

3. Both models are evaluated with the same background, so the
   FAR axis is directly comparable.

4. Verbose debugging: prints score distributions and sanity checks
   before producing plots.
"""

import argparse
import os
import logging

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SECONDS_PER_MONTH = 30 * 24 * 3600
SECONDS_PER_DAY   = 86400


# ──────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────

def load_predictions(hdf_path):
    """
    Load scores (and optionally distances) from an HDF5 predictions file.

    Expected datasets (any subset is ok):
        /scores       : float array — P(signal) per window
        /labels       : int array   — 1=signal, 0=noise  (optional)
        /distances    : float array — source distance in Mpc (optional)
        /duration     : scalar      — background duration in seconds (optional)
    """
    data = {}
    with h5py.File(hdf_path, 'r') as f:
        for key in ['scores', 'labels', 'distances', 'duration', 'probabilities', 'probs']:
            if key in f:
                val = f[key][()]
                # normalise key names
                if key in ('probabilities', 'probs'):
                    data['scores'] = val
                else:
                    data[key] = val
        # Some files store duration as an attribute
        if 'duration' not in data and 'duration' in f.attrs:
            data['duration'] = float(f.attrs['duration'])
    return data


# ──────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────

def compute_far_threshold_map(noise_scores, bg_duration_s):
    """
    Returns a function: FAR_per_month(threshold) → float.
    Also returns sorted_noise for fast lookup.
    """
    sorted_noise_desc = np.sort(noise_scores)[::-1]   # descending

    def far_pm(thr):
        n_above = (noise_scores >= thr).sum()
        return (n_above / bg_duration_s) * SECONDS_PER_MONTH

    return far_pm, sorted_noise_desc


def threshold_at_far(sorted_noise_desc, bg_duration_s, far_target_per_month):
    """Return the minimum score threshold that gives FAR ≤ target."""
    target_per_s = far_target_per_month / SECONDS_PER_MONTH
    n_allowed = int(np.floor(target_per_s * bg_duration_s))
    n_noise = len(sorted_noise_desc)
    if n_allowed < 1:
        return np.inf            # impossibly tight FAR for this background
    if n_allowed >= n_noise:
        return 0.0               # every noise sample passes → FAR saturates
    return float(sorted_noise_desc[n_allowed])


def sensitivity_curve(signal_scores, signal_distances, noise_scores,
                      bg_duration_s, percentile=50,
                      far_grid_per_month=None, n_points=200):
    """
    Sweep threshold from loose → tight and record (FAR_pm, D_p).

    Parameters
    ----------
    signal_scores     : P(signal) for each signal window
    signal_distances  : source distance in Mpc for each signal window
    noise_scores      : P(signal) for each noise window
    bg_duration_s     : background duration in seconds
    percentile        : which distance percentile (default 50 = median)
    far_grid_per_month: if provided, evaluate at these specific FAR values
    n_points          : number of threshold points to sweep

    Returns
    -------
    far_pm_arr  : array of FAR values (events / month)
    d_p_arr     : array of D_p values (Mpc)
    efficiency  : detection efficiency at each FAR
    """
    # Threshold grid spanning from 1st to 99th percentile of all scores
    all_scores = np.concatenate([signal_scores, noise_scores])
    thr_grid = np.percentile(all_scores, np.linspace(1, 99, n_points))

    far_pm_arr, d_p_arr, eff_arr = [], [], []
    for thr in thr_grid:
        # FAR (noise above threshold)
        far_ps = (noise_scores >= thr).sum() / bg_duration_s
        far_pm = far_ps * SECONDS_PER_MONTH

        # Detection efficiency + D_p
        found = signal_scores >= thr
        n_found = found.sum()
        if n_found == 0:
            continue
        eff = n_found / len(signal_scores)
        d_p = np.percentile(signal_distances[found], percentile) \
              if signal_distances is not None else eff

        far_pm_arr.append(far_pm)
        d_p_arr.append(d_p)
        eff_arr.append(eff)

    return np.array(far_pm_arr), np.array(d_p_arr), np.array(eff_arr)


# ──────────────────────────────────────────────────────────────────────────
# Debug / diagnostics
# ──────────────────────────────────────────────────────────────────────────

def print_score_diagnostics(name, signal_scores, noise_scores):
    """Print score distribution statistics to help diagnose problems."""
    print(f"\n  [{name}] Score diagnostics:")
    for label, sc in [("Signal", signal_scores), ("Noise", noise_scores)]:
        print(f"    {label:7}: "
              f"min={sc.min():.4f}  max={sc.max():.4f}  "
              f"mean={sc.mean():.4f}  std={sc.std():.4f}  "
              f"n={len(sc):,}")

    if signal_scores.std() < 0.01 and noise_scores.std() < 0.01:
        print("    ⚠  BOTH distributions near-constant — model output is saturated!")
        print("       Likely cause: whitening mismatch between training and evaluation,")
        print("       OR aux_clf/XGBoost was not trained (random init).")

    overlap = np.mean(signal_scores < noise_scores.mean())
    print(f"    Fraction of signals BELOW noise mean: {overlap:.3f}  "
          f"(should be < 0.3 for a useful detector)")


def plot_score_histograms(models_dict, save_path):
    """Histogram of scores per model, signal vs noise."""
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for col, (name, (sig_sc, noi_sc)) in enumerate(models_dict.items()):
        ax = axes[0][col]
        bins = np.linspace(0, 1, 80)
        ax.hist(noi_sc, bins=bins, alpha=0.6, color='royalblue',
                density=True, label=f'Noise (n={len(noi_sc):,})')
        ax.hist(sig_sc, bins=bins, alpha=0.6, color='crimson',
                density=True, label=f'Signal (n={len(sig_sc):,})')
        ax.set_yscale('log')
        ax.set(xlabel='P(signal)', ylabel='Density (log)', title=name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    fig.suptitle('Score distributions (debug)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  Score histograms → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

MODEL_STYLES = {
    'cnn': dict(color='steelblue',  lw=2, ls='-',  label='CNN-only'),
    'xgb': dict(color='darkorange', lw=2, ls='--', label='CNN+XGB'),
}


def plot_sensitivity_comparison(curves_dict, save_path, percentile=50,
                                far_lim=(1e-3, 1e4), d_lim=None):
    """
    Plot D_p vs FAR for multiple models on the same axes.

    curves_dict: {name: (far_pm_arr, d_p_arr)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: D_p vs FAR
    ax = axes[0]
    for name, (far_pm, d_p, eff) in curves_dict.items():
        s = MODEL_STYLES.get(name, dict(color='gray', lw=2, ls='-', label=name))
        order = np.argsort(far_pm)
        ax.semilogx(far_pm[order], d_p[order],
                    color=s['color'], lw=s['lw'], linestyle=s['ls'], label=s['label'])

    ax.set(xlabel='FAR (events / month)',
           ylabel=f'D{percentile} sensitive distance (Mpc)',
           title=f'Sensitivity — D{percentile} vs FAR',
           xlim=far_lim)
    if d_lim:
        ax.set_ylim(d_lim)
    ax.axvline(1.0,  color='gray', lw=1, ls=':', alpha=0.7, label='1/month')
    ax.axvline(0.1,  color='gray', lw=1, ls='--', alpha=0.5, label='0.1/month')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2, which='both')

    # Right: Efficiency vs FAR
    ax2 = axes[1]
    for name, (far_pm, d_p, eff) in curves_dict.items():
        s = MODEL_STYLES.get(name, dict(color='gray', lw=2, ls='-', label=name))
        order = np.argsort(far_pm)
        ax2.semilogx(far_pm[order], eff[order],
                     color=s['color'], lw=s['lw'], linestyle=s['ls'], label=s['label'])
    ax2.set(xlabel='FAR (events / month)',
            ylabel='Detection efficiency',
            title='Efficiency vs FAR',
            xlim=far_lim, ylim=[0, 1.05])
    ax2.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6, label='50% efficiency')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2, which='both')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓  Sensitivity comparison → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Correct FAR+D50 sensitivity analysis for CNN vs CNN+XGB.'
    )
    parser.add_argument('--foreground', required=True,
                        help='HDF5 with signal window scores (and optionally distances).')
    parser.add_argument('--background', required=True,
                        help='HDF5 with noise window scores.')
    parser.add_argument('--bg-duration', type=float, default=None,
                        help='Background data duration in seconds. '
                             'Read from file if not given.')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Sliding window step size in seconds (default 0.1).')
    parser.add_argument('--percentile', type=int, default=50,
                        help='Distance percentile for sensitivity (default 50).')
    parser.add_argument('--far-min', type=float, default=1e-3)
    parser.add_argument('--far-max', type=float, default=1e4)
    parser.add_argument('--output-plot', default='sensitivity_fixed.png')
    parser.add_argument('--output-debug', default='debug_scores.png')
    parser.add_argument('--output-data', default=None,
                        help='If set, save results to this HDF5 file.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING),
    )

    print("=" * 70)
    print("FIXED SENSITIVITY ANALYSIS  (FAR=window-based, D=D50)")
    print("=" * 70)

    # ── Load foreground (signal) predictions ──────────────────────────────
    print(f"\nLoading foreground: {args.foreground}")
    fg = load_predictions(args.foreground)
    if 'scores' not in fg:
        raise KeyError(f"No 'scores' dataset found in {args.foreground}.")
    signal_scores = np.asarray(fg['scores'], dtype=np.float64)
    signal_distances = np.asarray(fg.get('distances', np.ones_like(signal_scores) * np.nan))
    has_distances = not np.all(np.isnan(signal_distances))
    print(f"  Signal windows : {len(signal_scores):,}")
    if has_distances:
        print(f"  Distance range : {signal_distances[~np.isnan(signal_distances)].min():.1f} – "
              f"{signal_distances[~np.isnan(signal_distances)].max():.1f} Mpc")
    else:
        print("  Distances      : not available — sensitivity plot will show efficiency")
        signal_distances = np.ones_like(signal_scores)   # dummy

    # ── Load background (noise) predictions ───────────────────────────────
    print(f"Loading background: {args.background}")
    bg = load_predictions(args.background)
    if 'scores' not in bg:
        raise KeyError(f"No 'scores' dataset found in {args.background}.")
    noise_scores = np.asarray(bg['scores'], dtype=np.float64)
    print(f"  Noise windows  : {len(noise_scores):,}")

    # ── Background duration ────────────────────────────────────────────────
    if args.bg_duration is not None:
        bg_duration_s = args.bg_duration
    elif 'duration' in bg:
        bg_duration_s = float(bg['duration'])
        logging.info(f"Background duration read from file: {bg_duration_s}s")
    else:
        # Estimate: number of noise windows × step size
        bg_duration_s = len(noise_scores) * args.step_size
        logging.warning(f"bg-duration not provided — estimated as "
                        f"{len(noise_scores)} × {args.step_size}s = {bg_duration_s}s")

    print(f"  Background duration: {bg_duration_s:.0f}s "
          f"({bg_duration_s/SECONDS_PER_DAY:.3f} days)")

    # ── Score diagnostics ──────────────────────────────────────────────────
    print_score_diagnostics("model", signal_scores, noise_scores)

    # ── Debug score histogram ──────────────────────────────────────────────
    plot_score_histograms(
        {'model': (signal_scores, noise_scores)},
        save_path=args.output_debug,
    )

    # ── Sensitivity curve ──────────────────────────────────────────────────
    print("\nComputing sensitivity curve…")
    far_pm, d_p, eff = sensitivity_curve(
        signal_scores, signal_distances, noise_scores,
        bg_duration_s, percentile=args.percentile,
    )

    # ── Summary at standard FAR targets ───────────────────────────────────
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)
    sorted_noise_desc = np.sort(noise_scores)[::-1]
    for far_t in [0.01, 0.1, 1.0, 10.0]:
        thr = threshold_at_far(sorted_noise_desc, bg_duration_s, far_t)
        found = signal_scores >= thr
        n_found = found.sum()
        eff_t   = n_found / len(signal_scores)
        dp_t    = np.percentile(signal_distances[found], args.percentile) if n_found > 0 else 0.
        print(f"  FAR={far_t:8.2f}/month  thr={thr:.4f}  "
              f"efficiency={eff_t:.3f}  D{args.percentile}={dp_t:.1f}  "
              f"n_found={n_found:,}")

    # ── Plot ───────────────────────────────────────────────────────────────
    curves = {'xgb': (far_pm, d_p, eff)}   # rename if you have multiple models
    plot_sensitivity_comparison(
        curves, save_path=args.output_plot,
        percentile=args.percentile,
        far_lim=(args.far_min, args.far_max),
    )

    # ── Save data ──────────────────────────────────────────────────────────
    if args.output_data:
        with h5py.File(args.output_data, 'w') as f:
            f.create_dataset('far_per_month', data=far_pm)
            f.create_dataset(f'd{args.percentile}_mpc',  data=d_p)
            f.create_dataset('efficiency',    data=eff)
            f.attrs['bg_duration_s'] = bg_duration_s
            f.attrs['percentile']    = args.percentile
        print(f"✓  Results saved → {args.output_data}")

    print("\n✓  Done!")


if __name__ == '__main__':
    main()
