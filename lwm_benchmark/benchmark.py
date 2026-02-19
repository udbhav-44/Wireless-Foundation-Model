
import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Shared LWM constants/utils
from lwm.input_preprocess import (
    DeepMIMO_data_gen,
    deepmimo_data_cleaning,
    label_gen,
    tokenizer,
)
from lwm.inference import create_raw_dataset
from lwm.utils import FCN, get_data_loaders, train_model

# Model Imports
from lwm.lwm_model import lwm as lwm_base
from lwm_ca.torch_pipeline import LWMWithPrepatchCA, channels_to_patches, ensure_ri_channels, add_complex_noise_ri
from lwm_axial.torch_pipeline_axial import LWMWithPrepatchAxial
from lwm_physics.lwm_physics_model import lwm_physics

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LWM models on LoS/NLoS classification.")
    parser.add_argument("--scenarios", nargs="+", default=["O1_3p5B"], help="DeepMIMO scenarios")
    parser.add_argument("--task", type=str, default="los", help="los or beam")
    parser.add_argument("--models", nargs="+", default=["base", "ca", "axial", "physics"], 
                        choices=["base", "ca", "axial", "physics"], help="Models to benchmark")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[], help="Paths to checkpoints in order of models argument")
    parser.add_argument("--input-types", nargs="+", default=["cls_emb", "channel_emb"], choices=["raw", "cls_emb", "channel_emb"])
    parser.add_argument("--split-ratios", nargs="+", type=float, default=[0.01, 0.005])
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-folder", type=str, default="/home/audbhav22/foundation_model/Dataset")
    parser.add_argument("--n-beams", type=int, default=64)
    parser.add_argument("--save-csv", type=str, default="results/benchmark_comparison.csv")
    return parser.parse_args()

def normalize_task(task):
    task_lower = task.strip().lower()
    if task_lower in {"los", "los/nlos", "los_nlos"}:
        return "LoS/NLoS Classification"
    if task_lower in {"beam", "beam_prediction"}:
        return "Beam Prediction"
    return task

def load_deepmimo_data(scenarios, dataset_folder):
    if isinstance(scenarios, str): scenarios = [scenarios]
    return [DeepMIMO_data_gen(name, dataset_folder=dataset_folder) for name in scenarios]

def stack_cleaned_channels(deepmimo_data):
    cleaned = [deepmimo_data_cleaning(dm) for dm in deepmimo_data]
    return np.vstack(cleaned)

def create_labels_from_data(task, scenarios, deepmimo_data, n_beams):
    labels = []
    for scenario_name, data in zip(scenarios, deepmimo_data):
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams, visualize=False))
    return torch.tensor(labels).long()

# Helper Dataset for Base Model
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (torch.tensor(item[0]).float(), 
                torch.tensor(item[1]).float(), 
                torch.tensor(item[2]).long())

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ... (rest of main setup) ...
    # Skip to Model Loop

    # Cache setup (preserved, not shown in replacement if targeting specific block, but replace_file_content needs context)
    # I will target the benchmark loop specifically.

# ... skipping lines ...

        # Extract Embeddings
        embeddings = None
        if model_name == 'base':
            # Optimize Base model loading
            print("Creating DataLoader for Base model...")
            dataset = BaseDataset(preprocessed_base_toks)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            emb_list = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    # Move to device
                    batch = [b.to(args.device) for b in batch]
                    input_ids, masked_tokens, masked_pos = batch
                    _, output = model(input_ids, masked_pos)
                    emb_list.append(output.cpu())
            embeddings = torch.cat(emb_list, dim=0)
            
        else:
def train_eval_classifier(embeddings, labels, input_type, split_ratios, n_trials, num_classes, args):
    results = {} # split -> (mean, std)
    
    # Select features
    if input_type == 'cls_emb':
        X = embeddings[:, 0]
    elif input_type == 'channel_emb':
        X = embeddings[:, 1:]
        X = X.reshape(X.shape[0], -1) # Flatten sequence
    elif input_type == 'raw':
        X = embeddings.reshape(embeddings.shape[0], -1)
        
    f1_scores_matrix = np.zeros((n_trials, len(split_ratios)))
    
    for i, ratio in enumerate(split_ratios):
        for trial in range(n_trials):
            # Simple manual split or use utility
            train_loader, test_loader = get_data_loaders(X, labels, batch_size=args.batch_size, split_ratio=ratio, num_workers=4, pin_memory=True)
            
            # Simple MLP/Linear Probe
            clf = FCN(input_dim=X.shape[1], num_classes=num_classes)
            _, scores = train_model(clf, train_loader, test_loader, epochs=args.epochs, lr=1e-3, device=args.device)
            f1_scores_matrix[trial, i] = scores[0, -1]
            
        mean = f1_scores_matrix[:, i].mean()
        std = f1_scores_matrix[:, i].std()
        print(f"  Split {ratio}: F1 = {mean:.4f} +/- {std:.4f}")
        results[ratio] = (mean, std)
        
    return results

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Cache setup
    cache_dir = "lwm_benchmark/cache"
    os.makedirs(cache_dir, exist_ok=True)
    scenario_str = "_".join(sorted(args.scenarios))
    task_name = normalize_task(args.task)
    cache_file = os.path.join(cache_dir, f"{scenario_str}_{task_name.replace('/', '_')}_{args.n_beams}.pt")
    
    cleaned_channels = None
    labels = None
    preprocessed_base_toks = None
    
    loaded_from_cache = False
    if os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        try:
            cached_data = torch.load(cache_file, weights_only=False)
            cleaned_channels = cached_data['cleaned_channels']
            labels = cached_data['labels']
            preprocessed_base_toks = cached_data.get('preprocessed_base_toks') # Optional
            loaded_from_cache = True
            print("Data loaded successfully from cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating data.")
            
    if not loaded_from_cache:
        # 1. Load Data
        print(f"Loading scenarios: {args.scenarios}")
        deepmimo_data = load_deepmimo_data(args.scenarios, args.dataset_folder)
        cleaned_channels = stack_cleaned_channels(deepmimo_data)
        
        print(f"Generating labels for task: {task_name}")
        labels = create_labels_from_data(task_name, args.scenarios, deepmimo_data, args.n_beams).cpu()
        
        # Prepare Tokenized Data for Base Model (legacy support)
        if 'base' in args.models:
            print("Tokenizing data for Base model...")
            # Ensure (N, 32, 32) input for tokenizer (it expands dim 1 internally)
            channels_for_tok = cleaned_channels
            if channels_for_tok.ndim == 4 and channels_for_tok.shape[1] == 1:
                channels_for_tok = channels_for_tok.squeeze(1)
                
            preprocessed_base_toks = tokenizer(
                selected_scenario_names=None,
                manual_data=channels_for_tok,
                gen_raw=True, 
                dataset_folder=args.dataset_folder
            )
        
        # Save to cache
        print(f"Saving data to cache: {cache_file}")
        torch.save({
            'cleaned_channels': cleaned_channels,
            'labels': labels,
            'preprocessed_base_toks': preprocessed_base_toks
        }, cache_file, pickle_protocol=4)

    # Prepare Channel RI for advanced models
    # (N, 2, 32, 32)
    channels_c = cleaned_channels
    if channels_c.ndim == 3: channels_c = np.expand_dims(channels_c, 1)
    channels_ri = np.stack([channels_c.real, channels_c.imag], axis=1).squeeze(2).astype(np.float32)

    num_classes = 2 if "LoS" in task_name else args.n_beams

    # 2. Benchmark Loop
    all_results = []
    
    model_ckpt_map = dict(zip(args.models, args.checkpoints))
    
    for model_name in args.models:
        print(f"\n--- Benchmarking {model_name.upper()} ---")
        ckpt = model_ckpt_map.get(model_name)
        if not ckpt:
            print(f"Warning: No checkpoint provided for {model_name}, skipping.")
            continue
            
        model = load_model(model_name, ckpt, args.device)
        
        # Extract Embeddings
        embeddings = None
        if model_name == 'base':
            # Optimize Base model loading
            print("Creating DataLoader for Base model...")
            dataset = BaseDataset(preprocessed_base_toks)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            emb_list = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    # Move to device
                    batch = [b.to(args.device) for b in batch]
                    input_ids, masked_tokens, masked_pos = batch
                    _, output = model(input_ids, masked_pos)
                    emb_list.append(output.cpu())
            embeddings = torch.cat(emb_list, dim=0)
            
        else:
            # Physics, Axial, CA use the unified raw-channel extractor
            embeddings = extract_features(model_name, model, channels_ri, args.device, args.batch_size)
            
        # Train Classifiers
        for inp_type in args.input_types:
            print(f"Eval: {inp_type}")
            res = train_eval_classifier(embeddings, labels, inp_type, args.split_ratios, args.n_trials, num_classes, args)
            
            # Store results
            for ratio, (mean, std) in res.items():
                all_results.append({
                    'model': model_name,
                    'input_type': inp_type,
                    'split_ratio': ratio,
                    'f1_mean': mean,
                    'f1_std': std
                })

    # 3. Save Results
    if args.save_csv:
        print(f"Saving results to {args.save_csv}")
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'input_type', 'split_ratio', 'f1_mean', 'f1_std'])
            writer.writeheader()
            writer.writerows(all_results)

if __name__ == "__main__":
    main()
