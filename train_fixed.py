"""
train_fixed.py — CNN-XGBoost training with aux_clf saved
Key fix: saves aux_clf state dict so eval can load it for true CNN-only comparison.
"""

import os
import logging
import numpy as np
import h5py
import torch
import xgboost as xgb
from tqdm import tqdm
from argparse import ArgumentParser

# ── project imports ────────────────────────────────────────────────────────
from apply import get_base_network, dtype

dtype = torch.float32


# ──────────────────────────────────────────────────────────────────────────
# Dataset (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────
class Dataset(torch.utils.data.Dataset):
    def __init__(self, noises=None, waveforms=None,
                 store_device='cpu', train_device='cpu',
                 snr_range=(5., 15.)):
        super().__init__()
        self.noises = noises
        self.waveforms = waveforms
        self.store_device = store_device
        self.train_device = train_device
        if self.noises is not None:
            self.convert()
        self.rng = np.random.default_rng()
        self.snr_range = snr_range
        self.wave_label = torch.tensor([1., 0.]).to(dtype=dtype, device=train_device)
        self.noise_label = torch.tensor([0., 1.]).to(dtype=dtype, device=train_device)

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, i):
        if i < len(self.waveforms):
            snr = self.rng.uniform(*self.snr_range)
            return (self.noises[i] + snr * self.waveforms[i]).to(device=self.train_device), self.wave_label
        else:
            return self.noises[i].to(device=self.train_device), self.noise_label

    def convert(self):
        self.noises = torch.from_numpy(self.noises).to(dtype=dtype, device=self.store_device)
        self.waveforms = torch.from_numpy(self.waveforms).to(dtype=dtype, device=self.store_device)

    def save(self, h5py_file, group_name):
        if group_name in h5py_file:
            raise IOError(f"Group '{group_name}' already exists.")
        grp = h5py_file.create_group(group_name)
        grp.create_dataset('waveforms', data=self.waveforms.cpu().numpy())
        grp.create_dataset('noises', data=self.noises.cpu().numpy())

    def load(self, h5py_file, group_name):
        if group_name not in h5py_file:
            raise IOError(f"Group '{group_name}' does not exist.")
        grp = h5py_file[group_name]
        self.noises = grp['noises'][()]
        self.waveforms = grp['waveforms'][()]
        self.convert()


class reg_BCELoss(torch.nn.BCELoss):
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon * dim

    def forward(self, inputs, target, *args, **kwargs):
        assert inputs.shape[-1] == self.regularization_dim
        transformed = self.regularization_A + self.regularization_B * inputs
        return super().forward(transformed, target, *args, **kwargs)


def build_aux_classifier(embedding_dim, device):
    """Auxiliary classifier head: embeddings → 2-class softmax."""
    return torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 64),
        torch.nn.Dropout(p=0.3),
        torch.nn.ELU(),
        torch.nn.Linear(64, 2),
        torch.nn.Softmax(dim=1),
    ).to(dtype=dtype, device=device)


def train_cnn_feature_extractor(
        cnn, training_dataset, validation_dataset,
        output_dir, batch_size=32, learning_rate=5e-5,
        epochs=100, clip_norm=100, verbose=False,
        force=False, embedding_dim=128):
    """
    Stage 1: train CNN trunk + auxiliary classifier.
    CRITICAL FIX: saves both best_cnn.pt AND best_aux_clf.pt so the
    evaluation script can load the true CNN-only head for fair comparison.
    """
    logging.info("=" * 70)
    logging.info("STAGE 1: Training CNN Feature Extractor")
    logging.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    device = next(cnn.parameters()).device

    TrainDL = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    ValidDL = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=False)

    # ── auxiliary classifier (will be saved) ──────────────────────────────
    aux_clf = build_aux_classifier(embedding_dim, device)
    logging.info(f"Aux classifier: {embedding_dim} → 64 → 2")

    loss_fn = reg_BCELoss(dim=2)
    params = list(cnn.parameters()) + list(aux_clf.parameters())
    opt = torch.optim.Adam(params, lr=learning_rate)

    losses_path = os.path.join(output_dir, 'cnn_losses.txt')
    best_cnn_path = os.path.join(output_dir, 'best_cnn.pt')

    # ── FIX: path for aux classifier weights ──────────────────────────────
    best_aux_path = os.path.join(output_dir, 'best_aux_clf.pt')

    if os.path.isfile(losses_path) and not force:
        raise RuntimeError(f"Output file {losses_path} exists. Use --force to overwrite.")

    best_loss = 1e10
    with open(losses_path, 'w', buffering=1) as outfile:
        for epoch in tqdm(range(1, epochs + 1), desc="CNN training", disable=not verbose, ascii=True):
            cnn.train(); aux_clf.train()
            train_loss = 0.; train_batches = 0
            for x, y in tqdm(TrainDL, desc="  train", leave=False, disable=not verbose, ascii=True):
                opt.zero_grad()
                emb = cnn(x)
                out = aux_clf(emb)
                loss = loss_fn(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
                opt.step()
                train_loss += loss.item(); train_batches += 1

            cnn.eval(); aux_clf.eval()
            val_loss = 0.; val_batches = 0
            with torch.no_grad():
                for x, y in tqdm(ValidDL, desc="  valid", leave=False, disable=not verbose, ascii=True):
                    emb = cnn(x)
                    out = aux_clf(emb)
                    loss = loss_fn(out, y)
                    val_loss += loss.item(); val_batches += 1

            vl = val_loss / val_batches
            outfile.write(f"{epoch:04d}    {train_loss/train_batches:.6f}    {vl:.6f}\n")

            # Save per-epoch CNN checkpoint
            torch.save(cnn.state_dict(), os.path.join(output_dir, f'cnn_e{epoch:04d}.pt'))

            if vl < best_loss:
                best_loss = vl
                torch.save(cnn.state_dict(), best_cnn_path)
                # ── FIX: save aux_clf alongside CNN ───────────────────────
                torch.save(aux_clf.state_dict(), best_aux_path)
                logging.info(f"  Epoch {epoch}: best val_loss={best_loss:.6f}  → saved cnn + aux_clf")

    logging.info(f"Stage 1 complete. Best val loss: {best_loss:.6f}")
    return cnn, aux_clf


def extract_embeddings(cnn, dataset, batch_size=512, verbose=False):
    """Extract CNN embeddings for all samples. Returns (embeddings, binary_labels)."""
    cnn.eval()
    device = next(cnn.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting embeddings", disable=not verbose, ascii=True):
            embs.append(cnn(x.to(device)).cpu().numpy())
            labels.append(y[:, 0].cpu().numpy())   # [1,0]=signal → 1.0
    embeddings = np.vstack(embs)
    labels = np.concatenate(labels)

    # ── Debug: verify embeddings are non-trivial ───────────────────────────
    logging.info(f"Embedding shape: {embeddings.shape}")
    logging.info(f"Embedding mean: {embeddings.mean():.4f}  std: {embeddings.std():.4f}")
    logging.info(f"  (if std ≈ 0, CNN is not learning — check learning rate and data)")
    per_feature_std = embeddings.std(axis=0)
    dead_features = (per_feature_std < 1e-4).sum()
    if dead_features > 0:
        logging.warning(f"  {dead_features}/{embeddings.shape[1]} embedding dimensions are near-constant!")

    # ── Debug: check class separation in embedding space ──────────────────
    sig_emb = embeddings[labels > 0.5]
    noi_emb = embeddings[labels < 0.5]
    if len(sig_emb) > 0 and len(noi_emb) > 0:
        from numpy.linalg import norm
        sig_mean = sig_emb.mean(axis=0)
        noi_mean = noi_emb.mean(axis=0)
        separation = norm(sig_mean - noi_mean) / (embeddings.std() + 1e-8)
        logging.info(f"  Signal/noise centroid separation (normalised): {separation:.3f}")
        logging.info(f"  (should be > 1.0 for a useful embedding)")

    return embeddings, labels


def train_xgboost(train_emb, train_lbl, val_emb, val_lbl,
                  output_dir, verbose=False, force=False):
    """Stage 2: train XGBoost on CNN embeddings."""
    logging.info("=" * 70)
    logging.info("STAGE 2: Training XGBoost on CNN Embeddings")
    logging.info("=" * 70)
    logging.info(f"Train: {len(train_lbl)} samples  |  Val: {len(val_lbl)} samples")
    logging.info(f"Train positive rate: {train_lbl.mean():.3f}  Val: {val_lbl.mean():.3f}")

    # ── Debug: quick check that embeddings carry signal ───────────────────
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(train_emb, train_lbl.astype(int))
    lr_auc = roc_auc_score(val_lbl, lr.predict_proba(val_emb)[:, 1])
    logging.info(f"  Logistic Regression sanity AUC on val embeddings: {lr_auc:.4f}")
    if lr_auc < 0.6:
        logging.warning("  LR AUC < 0.6 — embeddings may not separate classes well!")
    elif lr_auc > 0.9:
        logging.info("  Embeddings look good — XGBoost should do at least this well.")

    xgb_params = dict(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
    )
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        train_emb, train_lbl.astype(int),
        eval_set=[(train_emb, train_lbl.astype(int)), (val_emb, val_lbl.astype(int))],
        verbose=50 if verbose else 0,
    )

    xgb_path = os.path.join(output_dir, 'best_xgboost.json')
    if os.path.isfile(xgb_path) and not force:
        raise RuntimeError(f"{xgb_path} exists. Use --force.")
    model.save_model(xgb_path)
    logging.info(f"XGBoost saved → {xgb_path}")

    val_auc = roc_auc_score(val_lbl, model.predict_proba(val_emb)[:, 1])
    logging.info(f"Final XGBoost val AUC: {val_auc:.4f}")
    return model


def train(cnn, training_dataset, validation_dataset, output_dir,
          batch_size=32, learning_rate=5e-5, epochs=100,
          clip_norm=100, verbose=False, force=False, embedding_dim=128):
    """Complete 3-stage training pipeline."""
    # Stage 1
    cnn, aux_clf = train_cnn_feature_extractor(
        cnn, training_dataset, validation_dataset, output_dir,
        batch_size=batch_size, learning_rate=learning_rate,
        epochs=epochs, clip_norm=clip_norm,
        verbose=verbose, force=force, embedding_dim=embedding_dim,
    )
    # Stage 2
    logging.info("Extracting training embeddings…")
    train_emb, train_lbl = extract_embeddings(cnn, training_dataset, batch_size=batch_size, verbose=verbose)
    logging.info("Extracting validation embeddings…")
    val_emb, val_lbl = extract_embeddings(cnn, validation_dataset, batch_size=batch_size, verbose=verbose)
    # Stage 3
    xgb_model = train_xgboost(train_emb, train_lbl, val_emb, val_lbl,
                               output_dir, verbose=verbose, force=force)

    logging.info("=" * 70)
    logging.info("TRAINING COMPLETE")
    logging.info(f"  {output_dir}/best_cnn.pt")
    logging.info(f"  {output_dir}/best_aux_clf.pt   ← NEW: needed for fair CNN-only eval")
    logging.info(f"  {output_dir}/best_xgboost.json")
    logging.info("=" * 70)
    return cnn, aux_clf, xgb_model


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = ArgumentParser(description="CNN-XGBoost training (fixed).")
    parser.add_argument('-d', '--dataset-file', nargs='+', required=True)
    parser.add_argument('-o', '--output-training', required=True)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--snr', type=float, nargs=2, default=(5., 15.))
    parser.add_argument('--weights')
    parser.add_argument('--coincident', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--clip-norm', type=float, default=100.)
    parser.add_argument('--train-device', default='cpu')
    parser.add_argument('--store-device', default='cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s | %(asctime)s: %(message)s',
        level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING),
        datefmt='%d-%m-%Y %H:%M:%S',
    )

    # Load datasets
    train_ds_list, val_ds_list = [], []
    for fpath in args.dataset_file:
        tds = Dataset(store_device=args.store_device, train_device=args.train_device, snr_range=args.snr)
        vds = Dataset(store_device=args.store_device, train_device=args.train_device, snr_range=args.snr)
        with h5py.File(fpath, 'r') as f:
            tds.load(f, 'training')
            vds.load(f, 'validation')
        train_ds_list.append(tds)
        val_ds_list.append(vds)

    train_ds = torch.utils.data.ConcatDataset(train_ds_list)
    val_ds   = torch.utils.data.ConcatDataset(val_ds_list)
    logging.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    num_det = train_ds_list[0].noises.shape[1]
    cnn = get_base_network(path=args.weights, device=args.train_device,
                           detectors=num_det, embedding_dim=args.embedding_dim)

    train(cnn, train_ds, val_ds, args.output_training,
          batch_size=args.batch_size, learning_rate=args.learning_rate,
          epochs=args.epochs, clip_norm=args.clip_norm,
          verbose=args.verbose, force=args.force, embedding_dim=args.embedding_dim)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    main()
