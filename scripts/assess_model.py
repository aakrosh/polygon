#!/usr/bin/env python
"""
Assess the quality of a trained POLYGON VAE model.

This script evaluates:
1. Validity: percentage of chemically valid SMILES
2. Uniqueness: percentage of unique molecules
3. Novelty: percentage not in training set
4. Reconstruction: ability to encode/decode molecules
5. Property distributions: MW, LogP, QED, etc.
"""

import sys
import argparse
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit import DataStructs
from collections import Counter
import torch

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib not installed. Property distribution plots will be skipped.")

# Add parent directory to path
sys.path.insert(0, '..')
from polygon.vae.vae_model import VAE
from polygon.utils.utils import load_model

def is_valid(smiles):
    """Check if SMILES string is valid."""
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_properties(smiles):
    """Calculate molecular properties."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'QED': QED.qed(mol)
    }

def assess_validity(smiles_list):
    """Calculate validity metrics."""
    valid = [is_valid(s) for s in smiles_list]
    validity = np.mean(valid)

    valid_smiles = [s for s, v in zip(smiles_list, valid) if v]

    return {
        'validity': validity,
        'valid_count': sum(valid),
        'total_count': len(smiles_list),
        'valid_smiles': valid_smiles
    }

def assess_uniqueness(smiles_list):
    """Calculate uniqueness metrics."""
    # Canonicalize SMILES (preserving stereochemistry)
    canonical = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    unique = list(set(canonical))
    uniqueness = len(unique) / len(canonical) if canonical else 0

    return {
        'uniqueness': uniqueness,
        'unique_count': len(unique),
        'total_valid': len(canonical),
        'unique_smiles': unique
    }

def assess_novelty(generated_smiles, training_smiles_file):
    """Calculate novelty (% not in training set).

    Note: generated_smiles should be pre-canonicalized with stereochemistry preserved.
    """
    # Load and canonicalize training set
    print("Loading training set...")
    with open(training_smiles_file, 'r') as f:
        training_smiles = set()
        for line in f:
            mol = Chem.MolFromSmiles(line.strip())
            if mol:
                training_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=True))

    print(f"Training set size: {len(training_smiles)}")

    # Check novelty
    novel = [s for s in generated_smiles if s not in training_smiles]
    novelty = len(novel) / len(generated_smiles) if generated_smiles else 0

    return {
        'novelty': novelty,
        'novel_count': len(novel),
        'total_unique': len(generated_smiles)
    }

def assess_internal_diversity(smiles_list, sample_size=1000):
    """Calculate internal diversity using Tanimoto similarity.

    Args:
        smiles_list: List of SMILES strings
        sample_size: Number of molecules to sample for pairwise comparison
                     (sampling used to avoid O(n^2) computation for large sets)

    Returns:
        Dictionary with diversity metrics
    """
    print(f"\nCalculating internal diversity...")

    # Sample if dataset is too large
    if len(smiles_list) > sample_size:
        print(f"Sampling {sample_size} molecules for diversity calculation...")
        sample = random.sample(smiles_list, sample_size)
    else:
        sample = smiles_list

    # Generate fingerprints
    fps = []
    for smi in sample:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fps.append(fp)

    if len(fps) < 2:
        return {
            'avg_tanimoto_similarity': 0.0,
            'avg_tanimoto_distance': 1.0,
            'molecules_compared': len(fps)
        }

    # Calculate pairwise Tanimoto similarities
    similarities = []
    n_comparisons = 0

    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
            n_comparisons += 1

    avg_similarity = np.mean(similarities)
    avg_distance = 1.0 - avg_similarity

    print(f"Compared {n_comparisons} molecule pairs")
    print(f"Average Tanimoto similarity: {avg_similarity:.3f}")
    print(f"Average Tanimoto distance (diversity): {avg_distance:.3f}")

    return {
        'avg_tanimoto_similarity': avg_similarity,
        'avg_tanimoto_distance': avg_distance,
        'molecules_compared': len(fps),
        'pairwise_comparisons': n_comparisons
    }

def assess_reconstruction(model, test_smiles, device, n_test=100):
    """Test model's ability to reconstruct molecules."""
    print(f"\nTesting reconstruction on {n_test} molecules...")

    model.eval()
    reconstructed = []
    original = []

    # Randomly sample to avoid selection bias (e.g., if data is sorted by property)
    n_test = min(n_test, len(test_smiles))
    test_subset = random.sample(test_smiles, n_test)

    for smiles in test_subset:
        try:
            # Encode using model.string2tensor to ensure proper multi-char token encoding
            x = model.string2tensor(smiles, device=device)

            # Get latent representation (use stochastic z, not deterministic mu)
            # This tests the true VAE reconstruction through the stochastic bottleneck
            z, kl_loss = model.forward_encoder([x])

            # Decode using sampled latent vector
            recon = model.sample(1, z=z, max_len=100)

            original.append(smiles)
            reconstructed.append(recon[0] if recon else '')
        except Exception as e:
            continue

    # Calculate semantic match rate (canonical SMILES comparison)
    exact_matches = 0
    valid_reconstructions = 0
    examples = []

    for o, r in zip(original, reconstructed):
        mol_o = Chem.MolFromSmiles(o)
        mol_r = Chem.MolFromSmiles(r) if r else None

        if mol_r:
            valid_reconstructions += 1

        if mol_o and mol_r:
            canon_o = Chem.MolToSmiles(mol_o, isomericSmiles=True)
            canon_r = Chem.MolToSmiles(mol_r, isomericSmiles=True)
            if canon_o == canon_r:
                exact_matches += 1

            # Collect first 5 examples
            if len(examples) < 5:
                examples.append({
                    'original': o,
                    'reconstructed': r,
                    'match': canon_o == canon_r
                })

    reconstruction_rate = exact_matches / len(original) if original else 0
    validity_rate = valid_reconstructions / len(original) if original else 0

    # Print examples
    print("\nReconstruction Examples:")
    for i, ex in enumerate(examples, 1):
        status = "✓ MATCH" if ex['match'] else "✗ DIFFER"
        print(f"  {i}. {status}")
        print(f"     Original:      {ex['original'][:60]}...")
        print(f"     Reconstructed: {ex['reconstructed'][:60]}...")

    return {
        'reconstruction_rate': reconstruction_rate,
        'exact_matches': exact_matches,
        'valid_reconstructions': valid_reconstructions,
        'validity_rate': validity_rate,
        'tested': len(original)
    }

def compare_property_distributions(gen_smiles, train_smiles_file, output_dir='./'):
    """Compare property distributions between generated and training sets."""
    print("\nComparing property distributions...")

    # Calculate properties for generated molecules
    gen_props = []
    for s in gen_smiles[:5000]:  # Sample for efficiency
        props = calculate_properties(s)
        if props:
            gen_props.append(props)

    # Calculate properties for training molecules
    print("Calculating training set properties (sampling 5000)...")
    train_props = []
    with open(train_smiles_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            props = calculate_properties(line.strip())
            if props:
                train_props.append(props)

    gen_df = pd.DataFrame(gen_props)
    train_df = pd.DataFrame(train_props)

    properties = ['MW', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'QED']

    # Create comparison plots (if matplotlib available)
    if HAS_PLOTTING:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, prop in enumerate(properties):
            if prop in gen_df.columns and prop in train_df.columns:
                axes[i].hist(train_df[prop], bins=50, alpha=0.5, label='Training', density=True)
                axes[i].hist(gen_df[prop], bins=50, alpha=0.5, label='Generated', density=True)
                axes[i].set_xlabel(prop)
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].set_title(f'{prop} Distribution')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/property_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved property distribution plot to {output_dir}/property_distributions.png")
    else:
        print("Skipping plots (matplotlib not available)")

    # Calculate KL divergence or statistics
    comparison_stats = {}
    for prop in properties:
        if prop in gen_df.columns and prop in train_df.columns:
            comparison_stats[prop] = {
                'train_mean': train_df[prop].mean(),
                'train_std': train_df[prop].std(),
                'gen_mean': gen_df[prop].mean(),
                'gen_std': gen_df[prop].std()
            }

    return comparison_stats

def main():
    parser = argparse.ArgumentParser(description='Assess POLYGON VAE model quality')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model.pt file')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training SMILES file')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of molecules to generate for assessment')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cpu or cuda:0)')
    parser.add_argument('--output', type=str, default='model_assessment.txt',
                       help='Output file for assessment results')
    parser.add_argument('--save_samples', action='store_true',
                       help='Save generated SMILES to file')

    args = parser.parse_args()

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(VAE, args.model_path, device)
    model.eval()
    print("Model loaded successfully!")

    # Generate molecules
    print(f"\nGenerating {args.n_samples} molecules...")
    generated_smiles = model.sample(args.n_samples, max_len=100)
    print(f"Generated {len(generated_smiles)} SMILES strings")

    if args.save_samples:
        with open('generated_samples.smi', 'w') as f:
            for s in generated_smiles:
                f.write(f"{s}\n")
        print("Saved generated SMILES to generated_samples.smi")

    # Run assessments
    print("\n" + "="*60)
    print("ASSESSMENT RESULTS")
    print("="*60)

    # 1. Validity
    print("\n1. VALIDITY ASSESSMENT")
    print("-" * 60)
    validity_results = assess_validity(generated_smiles)
    print(f"Valid SMILES: {validity_results['valid_count']}/{validity_results['total_count']}")
    print(f"Validity Rate: {validity_results['validity']:.2%}")

    # 2. Uniqueness
    print("\n2. UNIQUENESS ASSESSMENT")
    print("-" * 60)
    uniqueness_results = assess_uniqueness(validity_results['valid_smiles'])
    print(f"Unique molecules: {uniqueness_results['unique_count']}/{uniqueness_results['total_valid']}")
    print(f"Uniqueness Rate: {uniqueness_results['uniqueness']:.2%}")

    # 3. Novelty
    print("\n3. NOVELTY ASSESSMENT")
    print("-" * 60)
    novelty_results = assess_novelty(uniqueness_results['unique_smiles'], args.train_data)
    print(f"Novel molecules: {novelty_results['novel_count']}/{novelty_results['total_unique']}")
    print(f"Novelty Rate: {novelty_results['novelty']:.2%}")

    # 4. Internal Diversity
    print("\n4. INTERNAL DIVERSITY ASSESSMENT")
    print("-" * 60)
    diversity_results = assess_internal_diversity(uniqueness_results['unique_smiles'])
    print(f"Average Tanimoto similarity: {diversity_results['avg_tanimoto_similarity']:.3f}")
    print(f"Average Tanimoto distance (diversity): {diversity_results['avg_tanimoto_distance']:.3f}")
    print(f"Interpretation: Higher diversity (distance) is better (range: 0-1)")

    # 5. Reconstruction
    print("\n5. RECONSTRUCTION ASSESSMENT")
    print("-" * 60)
    with open(args.train_data, 'r') as f:
        test_smiles = [line.strip() for line in f.readlines()[:1000]]

    recon_results = assess_reconstruction(model, test_smiles, device, n_test=100)
    print(f"Exact reconstructions: {recon_results['exact_matches']}/{recon_results['tested']}")
    print(f"Reconstruction Rate (semantic): {recon_results['reconstruction_rate']:.2%}")
    print(f"Valid reconstructions: {recon_results['valid_reconstructions']}/{recon_results['tested']}")
    print(f"Reconstruction Validity: {recon_results['validity_rate']:.2%}")

    # 6. Property distributions
    print("\n6. PROPERTY DISTRIBUTION COMPARISON")
    print("-" * 60)
    prop_stats = compare_property_distributions(
        uniqueness_results['unique_smiles'],
        args.train_data,
        output_dir='.'
    )

    for prop, stats in prop_stats.items():
        print(f"\n{prop}:")
        print(f"  Training:  {stats['train_mean']:.2f} ± {stats['train_std']:.2f}")
        print(f"  Generated: {stats['gen_mean']:.2f} ± {stats['gen_std']:.2f}")

    # Save summary
    print("\n" + "="*60)
    print(f"Saving results to {args.output}...")

    with open(args.output, 'w') as f:
        f.write("POLYGON VAE MODEL ASSESSMENT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Training data: {args.train_data}\n")
        f.write(f"Samples generated: {args.n_samples}\n\n")

        f.write("METRICS SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Validity:       {validity_results['validity']:.2%}\n")
        f.write(f"Uniqueness:     {uniqueness_results['uniqueness']:.2%}\n")
        f.write(f"Novelty:        {novelty_results['novelty']:.2%}\n")
        f.write(f"Diversity:      {diversity_results['avg_tanimoto_distance']:.3f} (Tanimoto distance)\n")
        f.write(f"Reconstruction: {recon_results['reconstruction_rate']:.2%}\n\n")

        f.write("PROPERTY STATISTICS\n")
        f.write("-"*60 + "\n")
        for prop, stats in prop_stats.items():
            f.write(f"\n{prop}:\n")
            f.write(f"  Training:  {stats['train_mean']:.2f} ± {stats['train_std']:.2f}\n")
            f.write(f"  Generated: {stats['gen_mean']:.2f} ± {stats['gen_std']:.2f}\n")

    print(f"Assessment complete! Results saved to {args.output}")
    print("\nQUICK INTERPRETATION GUIDE:")
    print("- Validity should be >95% for a good model")
    print("- Uniqueness should be >80%")
    print("- Novelty: 50-90% is typically good (100% may indicate drift)")
    print("- Diversity: >0.7 Tanimoto distance indicates good chemical diversity")
    print("- Reconstruction >50% indicates good latent space learning")
    print("- Property distributions should be similar to training data")

if __name__ == '__main__':
    main()
