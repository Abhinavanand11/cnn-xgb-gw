#! /usr/bin/env python

# Modified to train CNN-XGBoost hybrid model

### Import modules
import os
print(os.environ.get("LD_LIBRARY_PATH"))
from argparse import ArgumentParser
import logging
import numpy as np
import h5py
pycbc.HAVE_CUDA = False
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector

import lal
import os, os.path
from tqdm import tqdm
import torch
import xgboost as xgb  # <<<< CHANGE 1: Added XGBoost import

from apply import get_coherent_network, get_coincident_network
from apply import dtype
import os
print(os.environ.get("LD_LIBRARY_PATH"))


### Basic dataset class for easy PyTorch loading
# [Dataset class remains UNCHANGED]
class Dataset(torch.utils.data.Dataset):
    def __init__(self, noises=None, waveforms=None,
                store_device='cpu', train_device='cpu',
                snr_range=(5., 15.)):
        torch.utils.data.Dataset.__init__(self)
        self.noises = noises
        self.waveforms = waveforms
        self.store_device = store_device
        self.train_device = train_device
        if not self.noises is None:
            self.convert()
        self.rng = np.random.default_rng()
        self.snr_range = snr_range
        self.wave_label = torch.tensor([1., 0.]).to(dtype=dtype, device=self.train_device)
        self.noise_label = torch.tensor([0., 1.]).to(dtype=dtype, device=self.train_device)
        return

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, i):
        if i<len(self.waveforms):
            snr = self.rng.uniform(*self.snr_range)
            return (self.noises[i]+snr*self.waveforms[i]).to(device=self.train_device), self.wave_label
        else:
            return self.noises[i].to(device=self.train_device), self.noise_label

    def convert(self):
        self.noises = torch.from_numpy(self.noises).to(dtype=dtype, device=self.store_device)
        self.waveforms = torch.from_numpy(self.waveforms).to(dtype=dtype, device=self.store_device)

    def save(self, h5py_file, group_name):
        if group_name in h5py_file.keys():
            raise IOError("Group '%s' in file already exists." % group_name)
        else:
            new_group = h5py_file.create_group(group_name)
            new_group.create_dataset('waveforms', data=self.waveforms.cpu().numpy())
            new_group.create_dataset('noises', data=self.noises.cpu().numpy())

    def load(self, h5py_file, group_name):
        if group_name in h5py_file.keys():
            group = h5py_file[group_name]
            self.noises = group['noises'][()]
            self.waveforms = group['waveforms'][()]
            self.convert()
        else:
            raise IOError("Group '%s' in file doesn't exist." % group_name)


class reg_BCELoss(torch.nn.BCELoss):
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon*self.regularization_dim
    def forward(self, inputs, target, *args, **kwargs):
        assert inputs.shape[-1]==self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B*inputs
        return torch.nn.BCELoss.forward(self, transformed_input, target, *args, **kwargs)


# <<<< CHANGE 2: New function to train CNN feature extractor
def train_cnn_feature_extractor(CNNNetwork, training_dataset, validation_dataset, 
                                 output_training, batch_size=32, learning_rate=5e-5, 
                                 epochs=100, clip_norm=100, verbose=False, 
                                 force=False, embedding_dim=128):
    """Train CNN to extract features (embeddings).
    
    This is Stage 1 of CNN-XGBoost training.
    The CNN learns to output meaningful embeddings via an auxiliary classifier.
    """
    logging.info("="*70)
    logging.info("STAGE 1: Training CNN Feature Extractor")
    logging.info("="*70)
    
    ### Set up data loaders
    logging.debug("Setting up datasets and data loaders.")
    TrainDL = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    ValidDL = torch.utils.data.DataLoader(validation_dataset, batch_size=500, shuffle=True)

    ### Create auxiliary classifier for CNN training
    # This classifier is only used during CNN training, not saved
    device = next(CNNNetwork.parameters()).device
    aux_classifier = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 64),
        torch.nn.Dropout(p=0.3),
        torch.nn.ELU(),
        torch.nn.Linear(64, 2),
        torch.nn.Softmax(dim=1)  # <<<< FIX: Add Softmax to output probabilities
    ).to(dtype=dtype, device=device)
    
    logging.debug(f"Created auxiliary classifier: {embedding_dim} -> 64 -> 2 (with Softmax)")

    ### Initialize loss function and optimizer
    logging.debug("Initializing loss function, optimizer and output file.")
    loss = reg_BCELoss(dim=2)
    
    # Optimize both CNN and auxiliary classifier
    params = list(CNNNetwork.parameters()) + list(aux_classifier.parameters())
    opt = torch.optim.Adam(params, lr=learning_rate)

    # <<<< FIX: Create output directory if it doesn't exist
    os.makedirs(output_training, exist_ok=True)
    
    losses_path = os.path.join(output_training, 'cnn_losses.txt')
    best_cnn_path = os.path.join(output_training, 'best_cnn.pt')
    
    if os.path.isfile(losses_path) and not force:
        raise RuntimeError("Output file %s exists." % losses_path)
    
    with open(losses_path, 'w', buffering=1) as outfile:
        ### Training loop
        best_loss = 1.e10
        for epoch in tqdm(range(1, epochs+1), desc="Training CNN", disable=not verbose, ascii=True):
            # Training epoch
            CNNNetwork.train()
            aux_classifier.train()
            training_running_loss = 0.
            training_batches = 0
            
            for training_samples, training_labels in tqdm(TrainDL, desc="Training batches", 
                                                         leave=False, disable=not verbose, ascii=True):
                opt.zero_grad()
                
                # CNN extracts embeddings
                embeddings = CNNNetwork(training_samples)
                
                # Auxiliary classifier predicts classes from embeddings
                training_output = aux_classifier(embeddings)
                
                training_loss = loss(training_output, training_labels)
                training_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
                
                opt.step()
                training_running_loss += training_loss.clone().cpu().item()
                training_batches += 1
            
            # Validation
            CNNNetwork.eval()
            aux_classifier.eval()
            
            with torch.no_grad():
                validation_running_loss = 0.
                validation_batches = 0
                
                for validation_samples, validation_labels in tqdm(ValidDL, desc="Validation", 
                                                                  leave=False, disable=not verbose, ascii=True):
                    embeddings = CNNNetwork(validation_samples)
                    validation_output = aux_classifier(embeddings)
                    validation_loss = loss(validation_output, validation_labels)
                    validation_running_loss += validation_loss.clone().cpu().item()
                    validation_batches += 1
            
            # Log and save
            validation_loss = validation_running_loss/validation_batches
            output_string = '%04i    %f    %f' % (epoch, training_running_loss/training_batches, validation_loss)
            outfile.write(output_string + '\n')
            
            # Save epoch checkpoint
            epoch_dict_path = os.path.join(output_training, 'cnn_state_dict_e_%04i.pt' % epoch)
            if not os.path.isfile(epoch_dict_path) or force:
                torch.save(CNNNetwork.state_dict(), epoch_dict_path)
            
            # Save best model
            if validation_loss < best_loss:
                torch.save(CNNNetwork.state_dict(), best_cnn_path)
                best_loss = validation_loss
                logging.info(f"Epoch {epoch}: New best CNN model saved (val_loss={best_loss:.6f})")

        logging.info(f"CNN training complete! Best validation loss: {best_loss:.6f}")
    
    return CNNNetwork


# <<<< CHANGE 3: New function to extract embeddings
def extract_embeddings(CNNNetwork, dataset, batch_size=500, verbose=False):
    """Extract CNN embeddings for all samples in dataset.
    
    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,) with binary labels (0=noise, 1=signal)
    """
    logging.info("Extracting CNN embeddings from dataset...")
    
    CNNNetwork.eval()
    device = next(CNNNetwork.parameters()).device
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for samples, labels in tqdm(dataloader, desc="Extracting embeddings", 
                                   disable=not verbose, ascii=True):
            samples = samples.to(device)
            
            # Extract embeddings from CNN
            embeddings = CNNNetwork(samples)
            
            # Convert to numpy and store
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Convert labels: [1,0] -> 1 (signal), [0,1] -> 0 (noise)
            binary_labels = labels[:, 0].cpu().numpy()
            all_labels.append(binary_labels)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    logging.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    logging.info(f"Signal samples: {labels.sum()}, Noise samples: {(1-labels).sum()}")
    
    return embeddings, labels


# <<<< CHANGE 4: New function to train XGBoost
def train_xgboost(train_embeddings, train_labels, val_embeddings, val_labels,
                  output_training, verbose=False, force=False):
    """Train XGBoost classifier on CNN embeddings.
    
    This is Stage 2 of CNN-XGBoost training.
    """
    logging.info("="*70)
    logging.info("STAGE 2: Training XGBoost on CNN Embeddings")
    logging.info("="*70)
    
    logging.info(f"Training samples: {len(train_labels)}")
    logging.info(f"Validation samples: {len(val_labels)}")
    logging.info(f"Embedding dimension: {train_embeddings.shape[1]}")
    logging.info(f"Train - Signal: {train_labels.sum()}, Noise: {(1-train_labels).sum()}")
    logging.info(f"Val   - Signal: {val_labels.sum()}, Noise: {(1-val_labels).sum()}")
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 30,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    logging.info("\nXGBoost parameters:")
    for key, val in xgb_params.items():
        logging.info(f"  {key}: {val}")
    
    # Train XGBoost
    logging.info("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    xgb_model.fit(
        train_embeddings,
        train_labels,
        eval_set=[(train_embeddings, train_labels), (val_embeddings, val_labels)],
        verbose=20 if verbose else 0
    )
    
    # Save model
    # <<<< FIX: Ensure directory exists
    os.makedirs(output_training, exist_ok=True)
    
    xgb_path = os.path.join(output_training, 'best_xgboost.json')
    if os.path.isfile(xgb_path) and not force:
        raise RuntimeError(f"Output file {xgb_path} exists.")
    
    xgb_model.save_model(xgb_path)
    logging.info(f"\nXGBoost model saved to: {xgb_path}")
    
    # Print final metrics
    if hasattr(xgb_model, 'best_iteration'):
        logging.info(f"Best iteration: {xgb_model.best_iteration}")
    
    evals_result = xgb_model.evals_result()
    if 'validation_1' in evals_result and 'auc' in evals_result['validation_1']:
        best_auc = max(evals_result['validation_1']['auc'])
        final_auc = evals_result['validation_1']['auc'][-1]
        logging.info(f"Best validation AUC: {best_auc:.4f}")
        logging.info(f"Final validation AUC: {final_auc:.4f}")
    
    return xgb_model


# <<<< CHANGE 5: Modified main train function
def train(CNNNetwork, training_dataset, validation_dataset, output_training,
          batch_size=32, learning_rate=5e-5, epochs=100,
          clip_norm=100, verbose=False, force=False, embedding_dim=128):
    """Complete CNN-XGBoost training pipeline.
    
    Stage 1: Train CNN to extract embeddings
    Stage 2: Extract embeddings from all data
    Stage 3: Train XGBoost on embeddings
    
    Returns
    -------
    CNNNetwork : trained CNN feature extractor
    xgb_model : trained XGBoost model
    """
    
    # Stage 1: Train CNN feature extractor
    CNNNetwork = train_cnn_feature_extractor(
        CNNNetwork, training_dataset, validation_dataset, output_training,
        batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
        clip_norm=clip_norm, verbose=verbose, force=force, 
        embedding_dim=embedding_dim
    )
    
    # Stage 2: Extract embeddings
    train_embeddings, train_labels = extract_embeddings(
        CNNNetwork, training_dataset, batch_size=batch_size, verbose=verbose
    )
    
    val_embeddings, val_labels = extract_embeddings(
        CNNNetwork, validation_dataset, batch_size=batch_size, verbose=verbose
    )
    
    # Stage 3: Train XGBoost
    xgb_model = train_xgboost(
        train_embeddings, train_labels, val_embeddings, val_labels,
        output_training, verbose=verbose, force=force
    )
    
    logging.info("="*70)
    logging.info("CNN-XGBOOST TRAINING COMPLETE!")
    logging.info("="*70)
    logging.info(f"Saved files:")
    logging.info(f"  - {output_training}/best_cnn.pt (CNN weights)")
    logging.info(f"  - {output_training}/best_xgboost.json (XGBoost model)")
    logging.info("="*70)
    
    return CNNNetwork, xgb_model


def main():
    parser = ArgumentParser(description="CNN-XGBoost training script for MLGWSC-1.")

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")

    parser.add_argument('-d', '--dataset-file', type=str, nargs='+', 
                       help="Path to the file where the datasets are stored.")
    parser.add_argument('-o', '--output-training', type=str, 
                       help="Path to the directory where the outputs will be stored.")
    parser.add_argument('-s', '--snr', type=float, nargs=2, default=(5., 15.), 
                       help="Range from which the optimal SNRs will be drawn. Default: 5. 15.")
    parser.add_argument('-w', '--weights', 
                       help="The path to the file containing initial CNN weights. Optional.")
    
    # <<<< CHANGE 6: Added embedding_dim parameter
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help="Dimension of CNN embeddings. Default: 128")
    
    parser.add_argument('--coincident', action='store_true', 
                       help="Train coincident search. Default: False (coherent)")
    parser.add_argument('--learning-rate', type=float, default=1e-5, 
                       help="Learning rate of the optimizer. Default: 0.00001")
    parser.add_argument('--epochs', type=int, default=100, 
                       help="Number of training epochs. Default: 100")
    parser.add_argument('--batch-size', type=int, default=32, 
                       help="Batch size of the training algorithm. Default: 32")
    parser.add_argument('--clip-norm', type=float, default=100., 
                       help="Gradient clipping norm. Default: 100.")
    parser.add_argument('--train-device', type=str, default='cpu', 
                       help="Device to train the network. Default: cpu")
    parser.add_argument('--store-device', type=str, default='cpu', 
                       help="Device to store the datasets. Default: cpu")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', 
                       level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Load datasets
    logging.info("Loading datasets...")
    TrainDS_list = []
    ValidDS_list = []
    
    for infname in args.dataset_file:
        logging.debug("Loading datasets from %s." % infname)
        TrainDS = Dataset(store_device=args.store_device, train_device=args.train_device, 
                         snr_range=args.snr)
        ValidDS = Dataset(store_device=args.store_device, train_device=args.train_device, 
                         snr_range=args.snr)
        
        with h5py.File(infname, 'r') as dataset_file:
            TrainDS.load(dataset_file, 'training')
            ValidDS.load(dataset_file, 'validation')
        
        TrainDS_list.append(TrainDS)
        ValidDS_list.append(ValidDS)
    
    TrainDS = torch.utils.data.ConcatDataset(TrainDS_list)
    ValidDS = torch.utils.data.ConcatDataset(ValidDS_list)
    logging.info(f"Datasets loaded: {len(TrainDS)} training, {len(ValidDS)} validation samples")

    ### Initialize CNN network (without XGBoost - that's trained later)
    logging.info("Initializing CNN network...")
    
    # <<<< CHANGE 7: Import get_base_network instead of full network
    from apply import get_base_network
    
    num_detectors = TrainDS[0][0].shape[0]
    logging.info(f"Number of detectors: {num_detectors}")
    
    # Initialize just the CNN (base network)
    CNNNetwork = get_base_network(
        path=args.weights, 
        device=args.train_device, 
        detectors=num_detectors,
        embedding_dim=args.embedding_dim
    )
    
    logging.info(f"CNN initialized with {args.embedding_dim}-dim embeddings")
    
    ### Train CNN-XGBoost
    CNNNetwork, xgb_model = train(
        CNNNetwork, TrainDS, ValidDS, args.output_training,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        epochs=args.epochs, clip_norm=args.clip_norm,
        verbose=args.verbose, force=args.force,
        embedding_dim=args.embedding_dim
    )

if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
