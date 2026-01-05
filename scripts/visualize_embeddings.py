#!/usr/bin/env python
"""
Visualize compound embeddings with Murcko scaffold coloring.

This script:
1. Loads compounds from SMILES or CSV file
2. Encodes them using a trained VAE model to get 128D latent embeddings
3. Reduces embeddings to 2D using UMAP
4. Computes Murcko scaffolds for each compound
5. Creates interactive plotly scatter plot colored by scaffold
6. Saves as HTML file
"""

import sys
import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

# Optional imports with graceful degradation
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Local imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
from polygon.vae.vae_model import VAE
from polygon.utils.utils import load_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize compound embeddings with Murcko scaffold coloring',
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=52),
        epilog="""
Examples:
  # Basic usage
  python visualize_embeddings.py --model_path models/vae.pt --input_file data/compounds.smi

  # CSV input with custom output
  python visualize_embeddings.py --model_path models/vae.pt \\
      --input_file data/compounds.csv --csv_column smiles \\
      --output results/embeddings_viz.html

  # Advanced: customize UMAP and visualization
  python visualize_embeddings.py --model_path models/vae.pt \\
      --input_file data/compounds.smi --n_neighbors 30 --min_dist 0.2 \\
      --top_n_scaffolds 30 --point_size 3 --device cpu
        """
    )

    # Required I/O arguments
    required_io = parser.add_argument_group('Required I/O arguments')
    required_io.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained VAE model (.pt file)'
    )
    required_io.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input file (SMILES .smi/.txt or CSV with "smiles" column)'
    )

    # Optional I/O arguments
    optional_io = parser.add_argument_group('Optional I/O arguments')
    optional_io.add_argument(
        '--output',
        type=str,
        default='embeddings_umap.html',
        help='Output HTML file path (default: embeddings_umap.html)'
    )
    optional_io.add_argument(
        '--csv_column',
        type=str,
        default='smiles',
        help='Column name for SMILES in CSV input (default: smiles)'
    )

    # Optional runtime arguments
    optional_runtime = parser.add_argument_group('Optional runtime arguments')
    optional_runtime.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (cpu or cuda:0, default: cuda:0)'
    )
    optional_runtime.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for encoding (default: 256)'
    )
    optional_runtime.add_argument(
        '--max_compounds',
        type=int,
        default=None,
        help='Maximum number of compounds to visualize (default: all)'
    )

    # UMAP parameters
    umap_params = parser.add_argument_group('UMAP parameters')
    umap_params.add_argument(
        '--n_neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter (default: 15)'
    )
    umap_params.add_argument(
        '--min_dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter (default: 0.1)'
    )
    umap_params.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='UMAP distance metric (default: euclidean)'
    )
    umap_params.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Visualization parameters
    viz_params = parser.add_argument_group('Visualization parameters')
    viz_params.add_argument(
        '--top_n_scaffolds',
        type=int,
        default=20,
        help='Number of top scaffolds to show separately (default: 20, others grouped as "Other")'
    )
    viz_params.add_argument(
        '--min_scaffold_size',
        type=int,
        default=5,
        help='Minimum compounds per scaffold to show separately (default: 5)'
    )
    viz_params.add_argument(
        '--point_size',
        type=int,
        default=5,
        help='Size of points in scatter plot (default: 5)'
    )
    viz_params.add_argument(
        '--width',
        type=int,
        default=1200,
        help='Plot width in pixels (default: 1200)'
    )
    viz_params.add_argument(
        '--height',
        type=int,
        default=800,
        help='Plot height in pixels (default: 800)'
    )

    return parser.parse_args()


def load_smiles_from_file(input_file, csv_column='smiles', max_compounds=None):
    """Load SMILES from .smi/.txt or .csv file.

    Args:
        input_file: Path to input file
        csv_column: Column name for CSV files
        max_compounds: Optional limit on number of compounds

    Returns:
        List of SMILES strings
    """
    logging.info(f"Loading SMILES from {input_file}...")

    file_ext = Path(input_file).suffix.lower()

    if file_ext in ['.smi', '.txt']:
        # Load SMILES file (one SMILES per line)
        with open(input_file, 'r') as handle:
            smiles_list = [line.rstrip() for line in handle.readlines()]
    elif file_ext == '.csv':
        # Load CSV and extract SMILES column
        df = pd.read_csv(input_file, sep=",", header=0)
        if csv_column not in df.columns:
            raise ValueError(f"Column '{csv_column}' not found in CSV. Available: {df.columns.tolist()}")
        smiles_list = df[csv_column].tolist()
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .smi, .txt, or .csv")

    # Remove empty/null entries
    smiles_list = [s for s in smiles_list if s and isinstance(s, str) and s.strip()]

    # Limit if requested
    if max_compounds is not None and len(smiles_list) > max_compounds:
        logging.info(f"Limiting to first {max_compounds} compounds (out of {len(smiles_list)})")
        smiles_list = smiles_list[:max_compounds]

    logging.info(f"Loaded {len(smiles_list)} SMILES strings")
    return smiles_list


def encode_smiles_to_embeddings(smiles_list, model, device, batch_size=256):
    """Encode SMILES to latent embeddings using VAE.

    Args:
        smiles_list: List of SMILES strings
        model: Trained VAE model
        device: torch device
        batch_size: Batch size for encoding

    Returns:
        embeddings: numpy array of shape (n_compounds, d_z)
        valid_smiles: list of valid SMILES (some may be filtered)
        valid_indices: indices of valid SMILES in original list
    """
    logging.info(f"Encoding {len(smiles_list)} compounds to latent space...")

    model.eval()
    collate_fn = model.get_collate_fn()

    embeddings_list = []
    valid_smiles = []
    valid_indices = []

    # Process in batches
    n_batches = (len(smiles_list) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(smiles_list))
        batch_smiles = smiles_list[start_idx:end_idx]

        # Filter valid SMILES in batch
        batch_valid = []
        batch_valid_indices = []
        for i, smi in enumerate(batch_smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    batch_valid.append(smi)
                    batch_valid_indices.append(start_idx + i)
            except Exception as e:
                logging.debug(f"Failed to parse SMILES '{smi}': {e}")
                pass

        if not batch_valid:
            continue

        try:
            # Encode batch
            with torch.no_grad():
                x_tensors = collate_fn(batch_valid)
                # Use encode() which returns mu (deterministic encoding)
                mu = model.encode(x_tensors)  # Shape: (batch_size, d_z)
                embeddings_list.append(mu.cpu().numpy())
                valid_smiles.extend(batch_valid)
                valid_indices.extend(batch_valid_indices)
        except Exception as e:
            logging.warning(f"Error encoding batch {batch_idx}: {e}")
            continue

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  Encoded {batch_idx + 1}/{n_batches} batches...")

    if not embeddings_list:
        raise ValueError("No compounds could be encoded successfully")

    embeddings = np.vstack(embeddings_list)

    n_invalid = len(smiles_list) - len(valid_smiles)
    if n_invalid > 0:
        logging.warning(f"Skipped {n_invalid} invalid SMILES")
    logging.info(f"Successfully encoded {len(valid_smiles)}/{len(smiles_list)} compounds")
    logging.info(f"Embedding shape: {embeddings.shape}")

    return embeddings, valid_smiles, valid_indices


def reduce_dimensions_umap(embeddings, n_neighbors=15, min_dist=0.1,
                           metric='euclidean', random_seed=42):
    """Reduce embeddings to 2D using UMAP.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_seed: Random seed for reproducibility

    Returns:
        embedding_2d: numpy array of shape (n_samples, 2)
    """
    if not HAS_UMAP:
        raise ImportError(
            "umap-learn is required for dimensionality reduction. "
            "Install with: uv sync --extra viz"
        )

    # Validate n_neighbors parameter
    n_samples = embeddings.shape[0]
    if n_neighbors >= n_samples:
        original_n_neighbors = n_neighbors
        n_neighbors = max(2, n_samples - 1)
        logging.warning(f"n_neighbors ({original_n_neighbors}) >= n_samples ({n_samples}). Adjusting to {n_neighbors}")

    logging.info(f"Reducing {embeddings.shape[1]}D embeddings to 2D using UMAP...")
    logging.info(f"  n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_seed,
        verbose=True
    )

    embedding_2d = reducer.fit_transform(embeddings)

    logging.info(f"UMAP reduction complete. Output shape: {embedding_2d.shape}")
    return embedding_2d


def compute_scaffolds(smiles_list, top_n=20, min_size=5):
    """Compute Murcko scaffolds for SMILES list.

    Args:
        smiles_list: List of SMILES strings
        top_n: Number of top scaffolds to show separately
        min_size: Minimum compounds per scaffold to show separately

    Returns:
        scaffold_labels: List of scaffold labels (same length as smiles_list)
        scaffold_smiles: List of scaffold SMILES (same length as smiles_list)
        scaffold_counts: Counter of scaffold frequencies
    """
    logging.info(f"Computing Murcko scaffolds for {len(smiles_list)} compounds...")

    scaffold_smiles = []
    for smi in smiles_list:
        try:
            scaffold = MurckoScaffoldSmilesFromSmiles(smi)
            # Handle empty scaffold (e.g., single atoms)
            if not scaffold or scaffold == '':
                scaffold = 'No_Scaffold'
        except Exception as e:
            logging.debug(f"Failed to compute scaffold for '{smi}': {e}")
            scaffold = 'Invalid'
        scaffold_smiles.append(scaffold)

    # Count scaffolds
    scaffold_counts = Counter(scaffold_smiles)

    logging.info(f"Found {len(scaffold_counts)} unique scaffolds")

    # Get top N most common scaffolds
    top_scaffolds = set([s for s, c in scaffold_counts.most_common(top_n)])

    # Also include scaffolds above minimum size threshold
    frequent_scaffolds = set([s for s, c in scaffold_counts.items() if c >= min_size])

    # Combine both criteria
    shown_scaffolds = top_scaffolds | frequent_scaffolds

    # Create labels (show top scaffolds individually, group rest as "Other")
    scaffold_labels = []
    for scaffold in scaffold_smiles:
        if scaffold in shown_scaffolds:
            # Label with scaffold and count
            count = scaffold_counts[scaffold]
            label = f"{scaffold} (n={count})"
        else:
            label = "Other"
        scaffold_labels.append(label)

    logging.info(f"Showing {len(shown_scaffolds)} scaffolds individually, rest grouped as 'Other'")

    # Count "Other" category
    n_other = sum(1 for label in scaffold_labels if label == "Other")
    if n_other > 0:
        logging.info(f"  'Other' category contains {n_other} compounds from {len(scaffold_counts) - len(shown_scaffolds)} scaffolds")

    return scaffold_labels, scaffold_smiles, scaffold_counts


def create_interactive_plot(embedding_2d, smiles_list, scaffold_labels,
                            scaffold_smiles, output_file,
                            point_size=5, width=1200, height=800):
    """Create interactive plotly scatter plot.

    Args:
        embedding_2d: 2D coordinates (n_samples, 2)
        smiles_list: List of SMILES strings
        scaffold_labels: List of scaffold labels for coloring
        scaffold_smiles: List of scaffold SMILES for tooltips
        output_file: Path to save HTML file
        point_size: Size of scatter points
        width: Plot width in pixels
        height: Plot height in pixels
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for visualization. "
            "Install with: uv sync --extra viz"
        )

    logging.info(f"Creating interactive plot...")

    # Create DataFrame for plotly
    df = pd.DataFrame({
        'UMAP_1': embedding_2d[:, 0],
        'UMAP_2': embedding_2d[:, 1],
        'SMILES': smiles_list,
        'Scaffold': scaffold_labels,
        'Scaffold_SMILES': scaffold_smiles,
    })

    # Sort by scaffold to ensure consistent legend ordering
    # Put "Other" at the end
    def sort_key(label):
        if label == "Other":
            return (1, label)  # Sort last
        else:
            # Extract count from "scaffold (n=X)" format
            try:
                count = int(label.split('(n=')[1].rstrip(')'))
                return (0, -count, label)  # Sort by count descending
            except:
                return (0, 0, label)

    df['_sort_key'] = df['Scaffold'].apply(lambda x: sort_key(x))
    df = df.sort_values('_sort_key')

    # Create color palette
    # Use plotly's built-in qualitative color sequences
    n_unique_scaffolds = df['Scaffold'].nunique()

    if n_unique_scaffolds <= 24:
        # Use Plotly's Dark24 palette for up to 24 categories
        colors = px.colors.qualitative.Dark24
    else:
        # Extend with other palettes if more categories
        colors = (px.colors.qualitative.Dark24 +
                 px.colors.qualitative.Light24 +
                 px.colors.qualitative.Alphabet)

    # Assign gray color to "Other" category
    color_map = {}
    scaffold_order = df['Scaffold'].unique()
    for i, scaffold in enumerate(scaffold_order):
        if scaffold == "Other":
            color_map[scaffold] = '#CCCCCC'  # Gray for "Other"
        else:
            color_map[scaffold] = colors[i % len(colors)]

    # Create scatter plot
    fig = go.Figure()

    for scaffold in scaffold_order:
        mask = df['Scaffold'] == scaffold
        subset = df[mask]

        fig.add_trace(go.Scatter(
            x=subset['UMAP_1'],
            y=subset['UMAP_2'],
            mode='markers',
            name=scaffold,
            marker=dict(
                size=point_size,
                color=color_map[scaffold],
                line=dict(width=0.5, color='white')
            ),
            text=[f"SMILES: {s}<br>Scaffold: {sc}"
                  for s, sc in zip(subset['SMILES'], subset['Scaffold_SMILES'])],
            hovertemplate='<b>%{text}</b><br>UMAP 1: %{x:.2f}<br>UMAP 2: %{y:.2f}<extra></extra>',
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Compound Embeddings Colored by Murcko Scaffold',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=width,
        height=height,
        hovermode='closest',
        legend=dict(
            title='Scaffold (count)',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10)
        ),
        template='plotly_white'
    )

    # Save to HTML
    fig.write_html(output_file)
    logging.info(f"Saved interactive plot to {output_file}")
    logging.info(f"  Plot dimensions: {width}x{height}")
    logging.info(f"  Points plotted: {len(df)}")
    logging.info(f"  Unique scaffolds: {n_unique_scaffolds}")


def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Parse arguments
    args = parse_args()

    # Check dependencies
    if not HAS_UMAP:
        logging.error("umap-learn is not installed. Install with: uv sync --extra viz")
        sys.exit(1)

    if not HAS_PLOTLY:
        logging.error("plotly is not installed. Install with: uv sync --extra viz")
        sys.exit(1)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Load model
    logging.info(f"Loading VAE model from {args.model_path}...")
    model = load_model(VAE, args.model_path, device)
    model.eval()
    logging.info(f"Model loaded. Latent dimension: {model.d_z}")

    # 2. Load SMILES
    smiles_list = load_smiles_from_file(
        args.input_file,
        csv_column=args.csv_column,
        max_compounds=args.max_compounds
    )

    # 3. Encode to embeddings
    embeddings, valid_smiles, valid_indices = encode_smiles_to_embeddings(
        smiles_list,
        model,
        device,
        batch_size=args.batch_size
    )

    # 4. Compute scaffolds
    scaffold_labels, scaffold_smiles, scaffold_counts = compute_scaffolds(
        valid_smiles,
        top_n=args.top_n_scaffolds,
        min_size=args.min_scaffold_size
    )

    # 5. UMAP dimensionality reduction
    embedding_2d = reduce_dimensions_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_seed=args.random_seed
    )

    # 6. Create interactive plot
    create_interactive_plot(
        embedding_2d,
        valid_smiles,
        scaffold_labels,
        scaffold_smiles,
        args.output,
        point_size=args.point_size,
        width=args.width,
        height=args.height
    )

    logging.info("Visualization complete!")
    logging.info(f"Open {args.output} in a web browser to view the interactive plot")


if __name__ == '__main__':
    main()
