#!/usr/bin/env python
"""
Prepare filtered ligand dataset for a given UniProt ID

Following the methodology from the POLYGON manuscript:
"We queried the Pharos GraphQL API and the BindingDB for small molecule
ligands against kinase proteins Q02750 and P42345 previously implicated
in human cancer. In concordance with the recommendations of the Pharos
web interface, we selected ligands with an IC50 concentration of less
than 1 µM against a given protein kinase target."

This script:
1. Queries BindingDB for a given UniProt ID
2. Filters for high-affinity binders (IC50 or Kd < 1 µM = 1000 nM)
3. Saves filtered dataset to CSV and SMILES files
"""

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
import logging
from pathlib import Path

def filter_ligands_for_target(
    uniprot_id,
    binding_db_path,
    output_dir='.',
    max_ic50_nm=1000,
    max_kd_nm=1000,
    save_format='both'
):
    """
    Filter BindingDB ligands for a target protein.

    Args:
        uniprot_id: UniProt ID of target protein (e.g., 'Q02750')
        binding_db_path: Path to BindingDB_All.tsv file
        output_dir: Directory to save output files
        max_ic50_nm: Maximum IC50 in nM (default 1000 nM = 1 µM)
        max_kd_nm: Maximum Kd in nM (default 1000 nM = 1 µM)
        save_format: 'csv', 'smiles', or 'both'

    Returns:
        DataFrame with filtered ligands
    """

    logging.info(f"Filtering BindingDB for target: {uniprot_id}")
    logging.info("Reading BindingDB line-by-line (memory efficient)...")

    # Read header to find column indices
    with open(binding_db_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')

    # Find indices of columns we need
    smiles_idx = header.index('Ligand SMILES')
    ic50_idx = header.index('IC50 (nM)')
    kd_idx = header.index('Kd (nM)')

    # Find all UniProt column indices
    uniprot_indices = [i for i, col in enumerate(header)
                      if 'UniProt (SwissProt) Primary ID of Target Chain' in col]

    logging.info(f"Searching across {len(uniprot_indices)} UniProt columns...")

    # Stream through file line-by-line
    matched_rows = []
    total_lines = 0

    with open(binding_db_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # Skip header

        for line_num, line in enumerate(f, 1):
            total_lines += 1

            # Progress indicator every 100k lines
            if line_num % 100000 == 0:
                logging.info(f"  Processed {line_num:,} lines, found {len(matched_rows):,} matches...")

            try:
                fields = line.strip().split('\t')

                # Check if target UniProt ID appears in any chain column
                uniprot_values = [fields[i] if i < len(fields) else '' for i in uniprot_indices]

                if uniprot_id in uniprot_values:
                    # Extract only the columns we need
                    smiles = fields[smiles_idx] if smiles_idx < len(fields) else ''
                    ic50 = fields[ic50_idx] if ic50_idx < len(fields) else ''
                    kd = fields[kd_idx] if kd_idx < len(fields) else ''

                    matched_rows.append({
                        'smiles': smiles,
                        'ic50': ic50,
                        'kd': kd
                    })
            except Exception as e:
                # Skip malformed lines
                continue

    logging.info(f"Scanned {total_lines:,} total lines")
    logging.info(f"Found {len(matched_rows):,} entries for {uniprot_id}")

    if len(matched_rows) == 0:
        logging.warning(f"No data found for UniProt ID: {uniprot_id}")
        return None

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(matched_rows)

    # Parse IC50 and Kd values (handle various formats)
    def parse_affinity(value):
        """Parse IC50/Kd values, handling >, <, and other prefixes."""
        if pd.isna(value) or value == '':
            return np.nan

        try:
            # Try direct float conversion first
            return float(value)
        except (ValueError, TypeError):
            # Handle values like '>10000' or '<0.5'
            try:
                value_str = str(value).strip()
                # Remove leading >, <, =, ~, etc.
                for prefix in ['>', '<', '=', '~', '≥', '≤']:
                    value_str = value_str.lstrip(prefix)
                return float(value_str)
            except (ValueError, TypeError):
                return np.nan

    df['ic50'] = df['ic50'].apply(parse_affinity)
    df['kd'] = df['kd'].apply(parse_affinity)

    # Count how many have IC50 or Kd data
    has_ic50 = df['ic50'].notna().sum()
    has_kd = df['kd'].notna().sum()
    has_either = ((df['ic50'].notna()) | (df['kd'].notna())).sum()

    logging.info(f"Entries with IC50 data: {has_ic50:,}")
    logging.info(f"Entries with Kd data: {has_kd:,}")
    logging.info(f"Entries with either: {has_either:,}")

    # Filter for high-affinity binders (IC50 < threshold OR Kd < threshold)
    # Following manuscript: "IC50 concentration of less than 1 µM"
    before_filter = len(df)

    # Keep if either IC50 < threshold OR Kd < threshold
    df = df[
        (df['ic50'] < max_ic50_nm) |
        (df['kd'] < max_kd_nm)
    ].copy()

    after_filter = len(df)
    logging.info(f"After filtering (IC50 < {max_ic50_nm} nM OR Kd < {max_kd_nm} nM): {after_filter:,} entries")
    logging.info(f"Removed {before_filter - after_filter:,} low-affinity ligands")

    if len(df) == 0:
        logging.warning(f"No high-affinity ligands found for {uniprot_id}")
        return None

    # Validate and canonicalize SMILES
    logging.info("Validating and canonicalizing SMILES...")
    valid_smiles = []
    valid_ic50 = []
    valid_kd = []
    invalid_count = 0

    for idx, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical_smiles)
            valid_ic50.append(row['ic50'])
            valid_kd.append(row['kd'])
        else:
            invalid_count += 1

    logging.info(f"Valid SMILES: {len(valid_smiles):,}")
    logging.info(f"Invalid SMILES removed: {invalid_count:,}")

    # Create clean dataframe
    df_clean = pd.DataFrame({
        'smiles': valid_smiles,
        'ic50_nm': valid_ic50,
        'kd_nm': valid_kd
    })

    # Remove duplicates (keep the one with best affinity)
    logging.info("Removing duplicate SMILES...")
    before_dedup = len(df_clean)

    # For duplicates, keep the one with the lowest IC50 or Kd
    df_clean['best_affinity'] = df_clean[['ic50_nm', 'kd_nm']].min(axis=1)
    df_clean = df_clean.sort_values('best_affinity').drop_duplicates(subset='smiles', keep='first')
    df_clean = df_clean.drop(columns=['best_affinity'])

    after_dedup = len(df_clean)
    logging.info(f"After removing duplicates: {after_dedup:,} unique ligands")
    logging.info(f"Removed {before_dedup - after_dedup:,} duplicates")

    # Sort by affinity
    df_clean = df_clean.sort_values('ic50_nm')

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_filename = f"{uniprot_id}_ligands_filtered"

    if save_format in ['csv', 'both']:
        csv_path = output_path / f"{base_filename}.csv"
        df_clean.to_csv(csv_path, index=False)
        logging.info(f"Saved CSV to: {csv_path}")

    if save_format in ['smiles', 'both']:
        smiles_path = output_path / f"{base_filename}.smiles"
        with open(smiles_path, 'w') as f:
            for smiles in df_clean['smiles']:
                f.write(f"{smiles}\n")
        logging.info(f"Saved SMILES to: {smiles_path}")

    # Print summary statistics
    logging.info("\n" + "="*60)
    logging.info("SUMMARY STATISTICS")
    logging.info("="*60)
    logging.info(f"UniProt ID: {uniprot_id}")
    logging.info(f"Total unique high-affinity ligands: {len(df_clean):,}")
    logging.info(f"Affinity threshold: IC50 < {max_ic50_nm} nM OR Kd < {max_kd_nm} nM")

    # IC50 statistics
    ic50_data = df_clean['ic50_nm'].dropna()
    if len(ic50_data) > 0:
        logging.info(f"\nIC50 statistics (n={len(ic50_data):,}):")
        logging.info(f"  Min: {ic50_data.min():.2f} nM")
        logging.info(f"  Median: {ic50_data.median():.2f} nM")
        logging.info(f"  Max: {ic50_data.max():.2f} nM")

    # Kd statistics
    kd_data = df_clean['kd_nm'].dropna()
    if len(kd_data) > 0:
        logging.info(f"\nKd statistics (n={len(kd_data):,}):")
        logging.info(f"  Min: {kd_data.min():.2f} nM")
        logging.info(f"  Median: {kd_data.median():.2f} nM")
        logging.info(f"  Max: {kd_data.max():.2f} nM")

    logging.info("="*60)

    return df_clean


def main():
    parser = argparse.ArgumentParser(
        description='Prepare filtered ligand dataset for a UniProt target',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter ligands for MEK1 (Q02750) with default 1 µM threshold
  python prepare_filtered_dataset.py --uniprot_id Q02750 --binding_db_path BindingDB_All.tsv

  # Use a stricter threshold (100 nM = 0.1 µM)
  python prepare_filtered_dataset.py --uniprot_id Q02750 --binding_db_path BindingDB_All.tsv --max_ic50 100

  # Process multiple targets
  python prepare_filtered_dataset.py --uniprot_id Q02750 --binding_db_path BindingDB_All.tsv
  python prepare_filtered_dataset.py --uniprot_id P42345 --binding_db_path BindingDB_All.tsv
        """
    )

    parser.add_argument('--uniprot_id', type=str, required=True,
                       help='UniProt ID of target protein (e.g., Q02750)')
    parser.add_argument('--binding_db_path', type=str, required=True,
                       help='Path to BindingDB_All.tsv file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--max_ic50', type=float, default=1000,
                       help='Maximum IC50 in nM (default: 1000 nM = 1 µM)')
    parser.add_argument('--max_kd', type=float, default=1000,
                       help='Maximum Kd in nM (default: 1000 nM = 1 µM)')
    parser.add_argument('--format', type=str, default='both',
                       choices=['csv', 'smiles', 'both'],
                       help='Output format (default: both)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        level=level,
        datefmt='%H:%M:%S'
    )

    # Run filtering
    df = filter_ligands_for_target(
        uniprot_id=args.uniprot_id,
        binding_db_path=args.binding_db_path,
        output_dir=args.output_dir,
        max_ic50_nm=args.max_ic50,
        max_kd_nm=args.max_kd,
        save_format=args.format
    )

    if df is not None:
        logging.info(f"\n✓ Successfully prepared filtered dataset for {args.uniprot_id}")
        logging.info(f"  Output files: {args.output_dir}/{args.uniprot_id}_ligands_filtered.*")
    else:
        logging.error(f"\n✗ Failed to prepare dataset for {args.uniprot_id}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
