#!/usr/bin/env python
"""
Hyperparameter optimization for delta-VAE + BoW using Optuna.

This script optimizes delta_target and lambda_bow to maximize both
validity and reconstruction rate for molecular VAE models.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna
import pandas as pd
import torch


def get_available_devices(n_jobs):
    """Get list of available devices for parallel trials.

    Returns a list of device strings (e.g., ['cuda:0', 'cuda:1', 'cpu']).
    Limits to one trial per GPU to prevent memory contention.
    """
    if not torch.cuda.is_available():
        print(f"CUDA not available, using CPU for all {n_jobs} workers")
        return ['cpu'] * n_jobs

    n_gpus = torch.cuda.device_count()

    # CRITICAL: Limit to one trial per GPU to prevent OOM
    if n_jobs > n_gpus:
        print(f"\nWARNING: n_jobs ({n_jobs}) > available GPUs ({n_gpus})")
        print(f"Reducing n_jobs to {n_gpus} to prevent GPU memory contention")
        print(f"Each VAE training can use 4-10GB GPU memory with batch_size=1024")
        n_jobs = n_gpus

    # Create device list with memory info
    devices = []
    print("\nGPU Memory Available:")
    for i in range(n_jobs):
        devices.append(f'cuda:{i}')
        try:
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            print(f"  cuda:{i}: {props.name} ({total_gb:.1f} GB)")
        except:
            pass

    return devices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize delta-VAE and BoW hyperparameters using Optuna"
    )

    # Required arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training SMILES file"
    )

    # Optuna settings
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="delta_bow_optimization",
        help="Name for the Optuna study"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db). If None, uses in-memory storage."
    )

    # Training settings
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=50,
        help="Number of epochs for each trial (use fewer than full training for speed)"
    )
    parser.add_argument(
        "--n_batch",
        type=int,
        default=1024,
        help="Batch size"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=50000,
        help="Use subset of training data for speed (0 = use all data)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of molecules to generate for assessment"
    )

    # Search space
    parser.add_argument(
        "--delta_min",
        type=float,
        default=0.01,
        help="Minimum delta_target to try"
    )
    parser.add_argument(
        "--delta_max",
        type=float,
        default=2.0,
        help="Maximum delta_target to try"
    )
    parser.add_argument(
        "--lambda_min",
        type=float,
        default=0.01,
        help="Minimum lambda_bow to try"
    )
    parser.add_argument(
        "--lambda_max",
        type=float,
        default=2.0,
        help="Maximum lambda_bow to try"
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="optuna_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for single-GPU training (ignored if --n_jobs > 1)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel trials (each uses a different GPU if available)"
    )

    # Other training parameters
    parser.add_argument(
        "--lr_start",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--clip_grad",
        type=int,
        default=50,
        help="Gradient clipping value"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def create_subset(train_data_path, subset_size, output_path, seed=42):
    """Create a random subset of training data using reservoir sampling.

    This method is memory-efficient and works with files of any size,
    only keeping subset_size lines in memory at once.
    """
    if subset_size <= 0:
        # Use all data
        return train_data_path

    random.seed(seed)

    # Use reservoir sampling for memory efficiency
    # This allows us to sample from arbitrarily large files
    reservoir = []

    with open(train_data_path, 'r') as f:
        # Fill reservoir with first subset_size lines
        for i, line in enumerate(f):
            if i < subset_size:
                reservoir.append(line)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < subset_size:
                    reservoir[j] = line

    # Check if we actually needed to sample
    if len(reservoir) < subset_size:
        print(f"Dataset has only {len(reservoir)} molecules, using all of them")
        return train_data_path

    # Write sampled subset to file
    with open(output_path, 'w') as f:
        f.writelines(reservoir)

    print(f"Sampled {len(reservoir)} molecules from dataset")
    return output_path


def assess_model(model_path, train_data, n_samples, device):
    """Run assessment script and return metrics."""
    # Use absolute path to assess_model.py
    script_dir = Path(__file__).parent
    assess_script = script_dir / "assess_model.py"

    # Create temporary file for JSON output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_json = f.name

    try:
        # Run assess_model.py
        result = subprocess.run([
            sys.executable,
            str(assess_script),
            "--model_path", model_path,
            "--train_data", train_data,
            "--n_samples", str(n_samples),
            "--device", device,
            "--output_json", output_json
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Assessment failed: {result.stderr}")
            return None

        # Read metrics from JSON
        with open(output_json, 'r') as f:
            metrics = json.load(f)

        # Validate metrics
        required_keys = ['validity', 'reconstruction_rate', 'uniqueness']
        missing = [k for k in required_keys if k not in metrics]
        if missing:
            print(f"Missing metrics in assessment output: {missing}")
            return None

        return metrics

    finally:
        # Clean up temporary file
        if os.path.exists(output_json):
            os.remove(output_json)


def objective(trial, args, train_data_subset, device=None):
    """Optuna objective function."""

    # Use provided device or fall back to args.device
    if device is None:
        device = args.device

    # Suggest hyperparameters
    delta_target = trial.suggest_float('delta_target', args.delta_min, args.delta_max, log=True)
    lambda_bow = trial.suggest_float('lambda_bow', args.lambda_min, args.lambda_max, log=True)
    smiles_augmentation = trial.suggest_categorical('smiles_augmentation', [True, False])

    # Create output directory for this trial
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    model_path = os.path.join(trial_dir, "model.pt")
    log_file = os.path.join(trial_dir, "training.log")

    # Build training command
    cmd = [
        sys.executable, "-m", "polygon.run", "train",
        "--train_data", train_data_subset,
        "--n_epoch", str(args.n_epoch),
        "--n_batch", str(args.n_batch),
        "--lr_start", str(args.lr_start),
        "--clip_grad", str(args.clip_grad),
        "--delta_target", str(delta_target),
        "--lambda_bow", str(lambda_bow),
        "--model_save", model_path,
        "--log_file", log_file,
        "--kl_start", "0",
        "--kl_w_start", "0.0",
        "--kl_w_end", "0.0",
        "--save_frequency", str(args.n_epoch),  # Only save final model
        "--device", device,
    ]

    if smiles_augmentation:
        cmd.append("--smiles_augmentation")

    # Run training
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: delta={delta_target:.4f}, lambda_bow={lambda_bow:.4f}, aug={smiles_augmentation}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Categorize error type for better debugging
            stderr_lower = result.stderr.lower()

            if "out of memory" in stderr_lower or "cuda error" in stderr_lower:
                error_type = "GPU_MEMORY"
                trial.set_user_attr('failure_type', error_type)
                print(f"GPU Memory Error in trial {trial.number}")
                # Clear cache for next trial
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
            elif "nan" in stderr_lower or "inf" in stderr_lower:
                error_type = "NUMERICAL_INSTABILITY"
                trial.set_user_attr('failure_type', error_type)
                print(f"Numerical instability in trial {trial.number}")
            else:
                error_type = "UNKNOWN"
                trial.set_user_attr('failure_type', error_type)

            # Store truncated error for debugging
            trial.set_user_attr('error_message', result.stderr[:500])
            print(f"Training failed ({error_type}): {result.stderr[:200]}")
            raise optuna.exceptions.TrialPruned()

        # Assess the model
        metrics = assess_model(model_path, train_data_subset, args.n_samples, device)

        if metrics is None:
            raise optuna.exceptions.TrialPruned()

        # Extract key metrics
        validity = metrics.get('validity', 0.0)
        reconstruction = metrics.get('reconstruction_rate', 0.0)
        uniqueness = metrics.get('uniqueness', 0.0)

        # Clear GPU cache to prevent memory fragmentation
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        # Store metrics as user attributes
        trial.set_user_attr('validity', validity)
        trial.set_user_attr('reconstruction_rate', reconstruction)
        trial.set_user_attr('uniqueness', uniqueness)

        # Compute objective: geometric mean of validity and reconstruction
        # This ensures both metrics must be good
        if validity > 0 and reconstruction > 0:
            score = (validity * reconstruction) ** 0.5
        else:
            score = 0.0

        print(f"Results: validity={validity:.3f}, reconstruction={reconstruction:.3f}, score={score:.3f}")

        return score

    except subprocess.TimeoutExpired:
        print("Trial timed out")
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.exceptions.TrialPruned()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate training data exists
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    # Create subset of training data if needed
    if args.subset_size > 0:
        print(f"Creating random subset of {args.subset_size} molecules from training data (seed={args.seed})...")
        subset_path = os.path.join(args.output_dir, "train_subset.smiles")
        train_data_subset = create_subset(args.train_data, args.subset_size, subset_path, seed=args.seed)
        print(f"Subset saved to: {subset_path}")
    else:
        train_data_subset = args.train_data

    # Create or load Optuna study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction='maximize',
            load_if_exists=True
        )
        print(f"Using persistent storage: {args.storage}")
    else:
        # Warn about in-memory storage with parallel trials
        if args.n_jobs > 1:
            print("\nWARNING: Using in-memory storage with parallel trials")
            print("For better reliability, consider using persistent storage:")
            storage_path = os.path.join(args.output_dir, "optuna.db")
            print(f"  --storage sqlite:///{storage_path}")
            print()

        study = optuna.create_study(
            study_name=args.study_name,
            direction='maximize'
        )

    # Set up device allocation for parallel trials
    if args.n_jobs > 1:
        devices = get_available_devices(args.n_jobs)
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"\n{'='*60}")
        print(f"Multi-GPU parallel optimization")
        print(f"Number of parallel workers: {args.n_jobs}")
        print(f"Available GPUs: {n_gpus}")
        print(f"Device assignment: {devices}")
        print(f"{'='*60}\n")

        # Create a device selector based on trial number
        def get_device_for_trial(trial):
            return devices[trial.number % len(devices)]

        def objective_with_device(trial):
            device = get_device_for_trial(trial)
            return objective(trial, args, train_data_subset, device=device)

        optimization_func = objective_with_device
    else:
        print(f"\n{'='*60}")
        print(f"Single-device optimization")
        print(f"Device: {args.device}")
        print(f"{'='*60}\n")
        optimization_func = lambda trial: objective(trial, args, train_data_subset)

    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Training epochs per trial: {args.n_epoch}")
    print(f"{'='*60}\n")

    # Run optimization
    study.optimize(
        optimization_func,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )

    # Print results
    print(f"\n{'='*60}")
    print("Optimization complete!")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest metrics:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results_file = os.path.join(args.output_dir, "optimization_results.json")
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_metrics': study.best_trial.user_attrs,
        'n_trials': len(study.trials)
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Save trials dataframe
    df = study.trials_dataframe()
    csv_file = os.path.join(args.output_dir, "trials.csv")
    df.to_csv(csv_file, index=False)
    print(f"Trial history saved to: {csv_file}")

    # Generate optimization history plot if optuna visualization is available
    try:
        import optuna.visualization as vis

        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(args.output_dir, "optimization_history.html"))

        # Plot parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(args.output_dir, "param_importances.html"))

        # Plot parallel coordinate
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(args.output_dir, "parallel_coordinate.html"))

        print(f"Visualizations saved to: {args.output_dir}/")
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")

    print(f"\n{'='*60}")
    print("Recommended training command with best hyperparameters:")
    print(f"{'='*60}")

    cmd = f"uv run polygon train \\\n"
    cmd += f"  --train_data {args.train_data} \\\n"
    cmd += f"  --n_epoch 200 \\\n"
    cmd += f"  --kl_start 0 \\\n"
    cmd += f"  --kl_w_start 0.0 \\\n"
    cmd += f"  --kl_w_end 0.0 \\\n"
    cmd += f"  --n_batch {args.n_batch} \\\n"
    cmd += f"  --lr_start {args.lr_start} \\\n"
    cmd += f"  --clip_grad {args.clip_grad} \\\n"
    cmd += f"  --delta_target {study.best_params['delta_target']:.6f} \\\n"
    cmd += f"  --lambda_bow {study.best_params['lambda_bow']:.6f} \\\n"
    if study.best_params.get('smiles_augmentation', False):
        cmd += f"  --smiles_augmentation \\\n"
    cmd += f"  --model_save best_model.pt \\\n"
    cmd += f"  --device {args.device}"

    print(cmd)
    print()


if __name__ == "__main__":
    main()
