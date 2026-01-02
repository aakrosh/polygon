# POLYpharmacology Generative Optimization Network (POLYGON) a VAE for de novo polypharmacology.

This repository contains the POLYGON framework, a de novo molecular generator for polypharmacology. Akin to de novo portait generation, POLYGON attempts to optimize the chemical space for multiple protein target domains.

![alt text](https://github.com/bpmunson/polygon/blob/main/images/Figure_1r.png?raw=true)

***

The codebase is primarily adapted from two excellent de novo molecular design frameworks:

1. GuacaMol for reward based reinforcement learning: https://github.com/BenevolentAI/guacamol 

2. MOSES for the VAE implementation: https://github.com/molecularsets/moses

## Data Sources
A key resource to the POLYGON framework is experimental binding data of small molecule ligands.  We use the BindingDB as a source for this information, which can be found here: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

Input molecule training datasets are available from the GuacaMol package:  https://github.com/BenevolentAI/guacamol 

## Installation of POLYGON

POLYGON requires Python 3.9 or higher (tested on 3.9.16).

### Using uv (Recommended)

Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and install POLYGON:
```bash
git clone https://github.com/bpmunson/polygon.git
cd polygon
uv sync
```

This will create a virtual environment and install all dependencies.

### For Development

To install with development dependencies:
```bash
uv sync --extra dev
```

### Legacy Installation (pip)

You can also install using pip:
```bash
git clone https://github.com/bpmunson/polygon.git
cd polygon
pip install .
```

Note: rdkit may require conda or system packages on some platforms.

### GPU Support

For GPU acceleration with PyTorch:
```bash
# Install CUDA-enabled PyTorch (adjust version as needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependencies

Core dependencies:
* pandas>=1.3.0
* numpy>=1.18.1
* rdkit>=2019.09.3
* torch>=1.4.0
* joblib>=0.14.1
* scikit-learn>=0.22.1

Installation time is on the order of minutes.

***


## Example Usage

### Using uv (Recommended)

Run commands directly with uv:
```bash
uv run polygon train --train_data ../data/guacamol_v1_train.smiles ...
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
polygon train --train_data ...
```

### Examples

Pretrain VAE to encode chemical embedding:
```bash
uv run polygon train \
	--train_data ../data/guacamol_v1_train.smiles \
	--log_file log.txt \
	--save_frequency 25 \
	--model_save model.pt \
	--n_epoch 200 \
	--n_batch 1024 \
	--debug \
	--d_dropout 0.2 \
	--device cpu
```

Train Ligand Binding Models for Two Protein Targets:
```bash
uv run polygon train_ligand_binding_model \
   --uniprot_id Q02750 \
   --binding_db_path BindingDB_All.csv \
   --output_path Q02750_ligand_binding.pkl
```

```bash
uv run polygon train_ligand_binding_model \
   --uniprot_id P42345 \
   --binding_db_path BindingDB_All.csv \
   --output_path P42345_ligand_binding.pkl
```

Use the chemical embedding to design polypharmacology compounds:
```bash
uv run polygon generate \
    --model_path ../data/pretrained_vae_model.pt \
    --scoring_definition scoring_definition.csv \
    --max_len 100 \
    --n_epochs 200 \
    --mols_to_sample 8192   \
    --optimize_batch_size 512    \
    --optimize_n_epochs 2   \
    --keep_top 4096   \
    --opti gauss   \
    --outF molecular_generation   \
    --device cpu  \
    --save_payloads   \
    --n_jobs 4 \
    --debug
```

The expected runtime for POLYGON is on the order of hours.

POLYGON will output designs as SMILES strings in a text file.  For example:
```
$ head GDM_final_molecules.txt
Fc1cc(F)cc(CC(Nc2ccc3ncccc3c2)c2cccnc2)c1
N[SH](=O)(O)c1cccc(S(=O)(=O)O)c1
N#Cc1cc(C(N)=NO)ccc1Nc1nccc2ccnn12
CN(CN=C(O)c1ccco1)Nc1nccs1
```
