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

def train_ligand_binding_model(target_unit_pro_id,binding_db_path,output_path):
    # Only read required columns
    # BindingDB has UniProt IDs for up to 50 target chains
    columns_needed = ['Ligand SMILES', 'IC50 (nM)', 'Kd (nM)']
    
    # Add UniProt columns for all possible chains (typically 1-10 are used)
    for i in range(1, 51):
        columns_needed.append(f'UniProt (SwissProt) Primary ID of Target Chain {i}')

    logging.info(f"Loading BindingDB data (only required columns)...")
    binddb = pd.read_csv(
        binding_db_path,
        sep="\t",
        header=0,
        usecols=lambda x: x in columns_needed,  # Use lambda to avoid errors if some columns don't exist
        low_memory=False,
        on_bad_lines='skip'
    )

    # Find rows where the target UniProt ID appears in ANY chain column
    uniprot_cols = [col for col in binddb.columns if 'UniProt (SwissProt) Primary ID of Target Chain' in col]
    mask = binddb[uniprot_cols].apply(lambda row: target_unit_pro_id in row.values, axis=1)

    d = binddb[mask]
    d = d[['Ligand SMILES','IC50 (nM)','Kd (nM)']]
    d.columns = ['smiles','ic50','kd50']

    logging.debug(f'Number of obs: {d.shape[0]}:')
    logging.debug(f'{d.head()}')

    # FIXED: Vectorized value parsing (handles NaN, empty strings, inequality operators)
    # Clean IC50 column
    d['ic50_str'] = d['ic50'].astype(str).str.strip().str.lstrip('<>=~')
    d['ic50_val'] = pd.to_numeric(d['ic50_str'], errors='coerce')

    # Clean Kd column
    d['kd_str'] = d['kd'].astype(str).str.strip().str.lstrip('<>=~')
    d['kd_val'] = pd.to_numeric(d['kd_str'], errors='coerce')

    # FIXED: Prioritize Kd over IC50 (Kd is direct binding measurement, IC50 is functional)
    d['metric_value'] = d['kd_val'].fillna(d['ic50_val'])

    # Convert to pKd/pIC50 and remove invalid values
    d = d[d['metric_value'] > 0]
    d['metric_value'] = -np.log10(d['metric_value'] * 1E-9)

    d = d[['smiles','metric_value']].dropna()
    d = d.drop_duplicates(subset='smiles')

    logging.debug(f'Number of obs: {d.shape[0]}:')

    if d.shape[0]<10:
        logging.info('Less than 10 compound-target pairs. Not fitting a model')
        return 1
    # convert to fingerprint
    fps = []
    values = []
    skipped = 0
    for x,y in d[['smiles','metric_value']].values:
        try:
            mol = Chem.MolFromSmiles(x)
            if mol is None:
                logging.debug(f'Invalid SMILES skipped: {x}')
                skipped += 1
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        except Exception as e:
            logging.debug(f'Failed to generate fingerprint for {x}: {e}')
            skipped += 1
            continue

        fps.append(fp)
        values.append(y)

    logging.info(f'Valid fingerprints: {len(fps)}, Invalid SMILES skipped: {skipped}')

    if len(fps) < 10:
        logging.warning('Less than 10 valid SMILES after fingerprinting')
        return 1

    X = np.array(fps)
    y = np.array(values)

    logging.info(f'Training Random Forest on {len(y)} samples')
    regr = RandomForestRegressor(n_estimators=1000,random_state=0,n_jobs=-1)
    regr.fit(X,y)

    score = regr.score(X,y)
    logging.info(f'Training R² score: {score:.3f}')

    if output_path is None:
        output_path = f'{target_unit_pro_id}_rfr_ligand_model.pt'

    with open(output_path, 'wb') as handle:
        s = pickle.dump(regr, handle)

    return 1
