# REVEAL++

**REVEAL++: Differentiable Phenotypic Grouping for Vision–Language Retinal Modeling of Alzheimer’s Disease Risk**

The model aligns retinal fundus images with structured clinical risk narratives to learn multimodal representations for incident Alzheimer’s disease prediction. Instead of assigning subjects to fixed phenotypic groups, REVEAL++ computes soft inter-subject similarity weights from retinal and clinical embeddings, enabling graded multi-positive contrastive learning. This continuous formulation better reflects the heterogeneous and spectrum-like nature of neurodegenerative disease risk and improves downstream prediction performance on UK Biobank retinal imaging data compared with discrete group-aware contrastive learning and standard vision–language baselines.


## Repository Contents

```text
REVEALPlusPlus/
├── README.md
├── LICENSE
├── requirements.txt
├── Stage1_Train_REVEAL.ipynb
├── Stage2_Downstream_Prediction.ipynb
├── run_optuna_assign.py
├── model.py
├── dataset.py
├── losses.py
├── utils.py
└── RETFound_MAE/
    ├── models_vit.py
    └── RETFound_cfp_weights.pth
```

### Main Files

| File | Purpose |
|---|---|
| `Stage1_Train_REVEAL.ipynb` | Stage 1 multimodal representation learning notebook |
| `Stage2_Downstream_Prediction.ipynb` | Stage 2 downstream AD prediction and evaluation notebook |
| `run_optuna_assign.py` | Optuna hyperparameter tuning script |
| `model.py` | REVEAL++ model wrapper using RETFound-MAE and GatorTron |
| `dataset.py` | Fundus image and clinical text dataset loader |
| `losses.py` | Contrastive and group-aware loss utilities |
| `utils.py` | Utility functions, including checkpoint and scheduler helpers |
| `requirements.txt` | Python package requirements |
| `LICENSE` | MIT license |

## Pretrained Checkpoints

Pretrained REVEAL++ checkpoints are hosted on Hugging Face:

```text
https://huggingface.co/smilelab/RevealPlusPlus/tree/main/checkpoints
```

Available checkpoint files:

| Checkpoint | Description |
|---|---|
| `revpp.pt` | REVEAL++ checkpoint |
| `gacl_true.pt` | REVEAL baseline with group-aware contrastive learning |
| `gacl_false.pt` | REVEAL baseline without group-aware contrastive learning |

Download checkpoints with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/smilelab/RevealPlusPlus
mkdir -p checkpoints
cp RevealPlusPlus/checkpoints/*.pt checkpoints/
```

## RETFound-MAE Setup

This repository expects the RETFound-MAE code and checkpoint to be available in the repository root:

```text
RETFound_MAE/
├── models_vit.py
└── RETFound_cfp_weights.pth
```

The model code imports RETFound-MAE modules directly:

```python
from RETFound_MAE import models_vit
from RETFound_MAE.models_vit import vit_large_patch16, VisionTransformer
```

The default RETFound checkpoint path used by the training code is:

```text
./RETFound_MAE/RETFound_cfp_weights.pth
```

Make sure `RETFound_MAE/` is present before running Stage 1 training or downstream evaluation.

## Installation

Clone the repository:

```bash
git clone https://github.com/lab-smile/RevealPlusPlus.git
cd RevealPlusPlus
```

Create and activate a conda environment:

```bash
conda create -n revealpp python=3.11 -y
conda activate revealpp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Raw UK Biobank data, retinal images, clinical variables, and protected health information are not included in this repository.

The dataset loader expects COCO-style JSON files containing image metadata and clinical text annotations, along with a CSV file containing retinal image features. Structured UK Biobank risk-factor variables were converted into synthetic clinical reports using a fixed template and LLaMA-3.1. The template is as follows

```python
The subject is <age>years old <ethnic background><sex>. The average total household of this subject is in between <economic status>. The subject has <HbA1C> HbA1C, <HDL> HDL, <BMI> BMI, <systolic blood pressure> systolic blood pressure, <diastolic blood pressure> diastolic blood pressure. For lifestyle, the subject is in <employment status>. The subject is <smoking history>, has <depression>, has sleep deprivation <sleep deprivation>, and drinks alcohol <alcohol use>. The subject had his first cannabis at age <age of cannabis initiation>and used cannabis <cannabis use>times. The subject visits family <frequency of family visit>, and <number of leisure activity>. For physical activity, the subject walks <duration of walked 10+ minutes>minutes <number of days/week of walked 10+ minutes>days per week, exercises moderately <duration of moderate activity>minutes for <number of days/week of moderate activity>days a week, and exercises vigorously <duration of vigorous exercise> minutes for <number of days/week of vigorous activity> days a week. For diet, the subject has <cooked vegetable intake> tablespoons of cooked vegetables, <raw vegetable intake> tablespoons of raw vegetables, <fresh fruit intake> tablespoons of fresh fruit, and <dried fruit intake> dried fruit. In addition, the subject has oily fish <oily fish intake>, non-oily fish <non oily fish intake>, processed meat <processed meat intake>, poultry <poultry intake>, beef <beef intake>, lamb <lamb intake>, and pork <pork intake>. The subject has <bread intake> slices of bread per week, with <spread type>. The subject drinks <milk type>, <tea intake>cups of tea, <coffee intake> cups of coffee, <water intake> cups of water per day. The subject puts <salt added to food> in his diet. For cognitive function, the subject remembered <numeric memory> digits in the numeric memory test, scored <fluid intelligence> in a fluid intelligence test, completed trail #1 in <trail-making test A duration> deciseconds with <trail-making test A error counts> errors, and completed trail #2 in <trail-making test B duration> deciseconds with <trail-making test B error counts> errors. 
```

When a risk factor was unavailable (e.g., age of cannabis initiation), the report stated: No cannabis use was reported at that age in the <age of cannabis initiation>section.

Default expected files:

```text
data/UKB/captions_train.json
data/UKB/captions_val.json
data/UKB/captions_test2.json
data/UKB/Macular_Measurement.csv
```

A typical JSON annotation structure should include:

```json
{
  "images": [
    {
      "id": "participant_id",
      "file_name": "participant_id.png",
      "path": "/path/to/fundus/image.png"
    }
  ],
  "annotations": [
    {
      "image_id": "participant_id",
      "caption": "Clinical risk narrative for the subject..."
    }
  ]
}
```

The dataset returns:

```python
image, text, image_features
```

where `image` is a transformed fundus image and `text` is a clinical narrative.

## Stage 1: Train REVEAL++

Open and run:

```text
Stage1_Train_REVEAL.ipynb
```

The Stage 1 notebook trains the multimodal image-text model and saves checkpoints to the configured run directory.

Key configuration fields include:

```python
config = OmegaConf.create({
    "saved_checkpoints": RUN_DIR,
    "image_feature_file": "data/UKB/Macular_Measurement.csv",
    "train_json": "captions_train.json",
    "val_json": "captions_val.json",
    "test_json": "captions_test2.json",
    "vision_model_checkpoint": "./RETFound_MAE/RETFound_cfp_weights.pth",
    "vision_model_output_dim": 1024,
    "text_model_output_dim": 1024,
    "context_length": 512,
    "proj_dim": 1024,
    "num_train_epochs": 2,
    "batch_size": 16,
    "gradient_accumulation_steps": 8,
    "loss": "mutual",
})
```

Update paths before running on your system.

## Hyperparameter Tuning

Run Optuna tuning with:

```bash
python run_optuna_assign.py --study-name retfound_study --n-trials 30
```

Optional dashboard:

```bash
optuna-dashboard --storage sqlite:///retfound_study.db --host 0.0.0.0 --port 5555
```

## Stage 2: Downstream Prediction

Open and run:

```text
Stage2_Downstream_Prediction.ipynb
```

The Stage 2 notebook extracts learned image-text representations and evaluates downstream Alzheimer’s disease prediction.

The evaluation reports metrics including:

- AUROC
- balanced accuracy
- F1-score
- Matthews correlation coefficient

## Reported Results

REVEAL++ achieved the strongest performance among the evaluated methods for incident Alzheimer’s disease prediction.

| Method | AUROC | Balanced Accuracy | F1-Score | MCC |
|---|---:|---:|---:|---:|
| Baseline SVM | 0.593 ± 0.068 | 0.574 ± 0.083 | 0.140 ± 0.089 | 0.076 ± 0.099 |
| KeepFIT-CFP | 0.490 ± 0.063 | 0.505 ± 0.041 | 0.099 ± 0.034 | 0.002 ± 0.046 |
| BiomedCLIP | 0.525 ± 0.064 | 0.522 ± 0.060 | 0.121 ± 0.052 | 0.023 ± 0.054 |
| RETCLIP | 0.558 ± 0.076 | 0.527 ± 0.042 | 0.106 ± 0.069 | 0.028 ± 0.051 |
| PMC-CLIP | 0.471 ± 0.049 | 0.484 ± 0.020 | 0.076 ± 0.023 | -0.022 ± 0.023 |
| RETFound + GatorTron | 0.642 ± 0.052 | 0.581 ± 0.069 | 0.185 ± 0.099 | 0.119 ± 0.101 |
| REVEAL, no GACL | 0.654 ± 0.092 | 0.602 ± 0.075 | 0.205 ± 0.096 | 0.144 ± 0.105 |
| REVEAL, with GACL | 0.658 ± 0.090 | 0.609 ± 0.079 | 0.207 ± 0.100 | 0.146 ± 0.111 |
| REVEAL++ | 0.678 ± 0.061 | 0.613 ± 0.048 | 0.236 ± 0.079 | 0.168 ± 0.088 |

## Citation

If you use this repository or pretrained checkpoints, please cite:

```bibtex
@inproceedings{meidinger2026revealplusplus,
  title     = {REVEAL++: Differentiable Phenotypic Grouping for Vision--Language Retinal Modeling of Alzheimer's Disease Risk},
  author    = {Meidinger, Ethan and Leem, Seowung and Zhao, Zeyun and Fang, Ruogu},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```

## License

This repository is released under the MIT License.

