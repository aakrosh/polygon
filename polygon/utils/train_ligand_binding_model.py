import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from pathlib import Path

import logging

def train_ligand_binding_model(target_unit_pro_id, binding_db_path, output_path):
    import gc

    columns_needed = ['Ligand SMILES', 'IC50 (nM)', 'Kd (nM)']
    for i in range(1, 51):
        columns_needed.append(f'UniProt (SwissProt) Primary ID of Target Chain {i}')

    logging.info("Loading BindingDB data in chunks...")

    chunksize = 50_000
    collected = []

    for chunk in pd.read_csv(
        binding_db_path,
        sep="\t",
        header=0,
        usecols=lambda x: x in columns_needed,
        low_memory=False,
        on_bad_lines='skip',
        chunksize=chunksize
    ):
        # Find rows where the target UniProt ID appears in ANY chain column
        uniprot_cols = [
            col for col in chunk.columns
            if 'UniProt (SwissProt) Primary ID of Target Chain' in col
        ]

        mask = chunk[uniprot_cols].apply(
            lambda row: target_unit_pro_id in row.values,
            axis=1
        )

        d_chunk = chunk[mask][['Ligand SMILES', 'IC50 (nM)', 'Kd (nM)']]
        d_chunk.columns = ['smiles', 'ic50', 'kd50']

        if not d_chunk.empty:
            collected.append(d_chunk)

        del chunk, d_chunk
        gc.collect()

    if not collected:
        logging.info("No matching compound-target pairs found")
        return 1

    d = pd.concat(collected, ignore_index=True)

    logging.debug(f'Number of obs: {d.shape[0]}:')
    logging.debug(f'{d.head()}')

    vs = []
    for i, j in d[['ic50', 'kd50']].values:
        try:
            v = float(i)
        except (ValueError, TypeError):
            try:
                v = float(str(i)[1:])
            except (ValueError, TypeError):
                v = np.nan

        try:
            w = float(j)
        except (ValueError, TypeError):
            try:
                w = float(str(j)[1:])
            except (ValueError, TypeError):
                w = np.nan

        t = pd.Series([v, w]).dropna().min()

        if pd.notna(t) and t > 0:
            t = -np.log10(t * 1E-9)
            vs.append(t)
        else:
            vs.append(np.nan)

    d['metric_value'] = vs
    d = d[['smiles', 'metric_value']]
    d['metric_value'] = d['metric_value'].astype(float)
    d = d.drop_duplicates(subset='smiles')
    d = d.dropna()

    logging.debug(f'Number of obs: {d.shape[0]}:')

    if d.shape[0] < 10:
        logging.info('Less than 10 compound-target pairs. Not fitting a model')
        return 1

    fps = []
    values = []
    invalid_smiles_count = 0

    for x, y in d[['smiles', 'metric_value']].values:
        try:
            mol = Chem.MolFromSmiles(x)
            if mol is None:
                invalid_smiles_count += 1
                continue
            fp = rdFingerprintGenerator.GetMorganGenerator(2).GetFingerprint(mol)
        except Exception as e:
            invalid_smiles_count += 1
            logging.debug(f"Failed to generate fingerprint for SMILES '{x}': {e}")
            continue

        fps.append(fp)
        values.append(y)

    if invalid_smiles_count > 0:
        logging.info(f"Skipped {invalid_smiles_count} molecules with invalid SMILES")

    X = np.array(fps)
    y = np.array(values)

    logging.info(f"Training ligand binding model with {len(X)} valid compound-target pairs")

    if len(X) < 10:
        logging.info('Less than 10 valid compound-target pairs after filtering. Not fitting a model')
        return 1

    if not np.all(np.isfinite(y)):
        logging.error(
            f"Target values contain {np.sum(~np.isfinite(y))} non-finite values. Cannot train model."
        )
        return 1

    regr = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    regr.fit(X, y)

    logging.debug(regr.score(X, y))

    if output_path is None:
        output_path = f'{target_unit_pro_id}_rfr_ligand_model.pt'

    with open(output_path, 'wb') as handle:
        pickle.dump(regr, handle)

    return 1
