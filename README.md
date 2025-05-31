# Privacy-Preserving Movie Recommendation System

This project shows the construction of a MovieLens 100k recommender system using
- Neural Collaborative Filtering (NCF) (Baseline),
- Differential Privacy via Opacus (DP-SGD),
- Differential Privacy via IBM DiffPrivLib (DP-LinearRegression).

Goal: Investigate and demonstrate privacy-utility trade-off.

## Installation

1. clone repository:
 git clone https://github.com/erionkr/privacy-preserving-recsys.git
 cd privacy-preserving-recsys

2. Create a virtual environment:
 python3 -m venv .venv
 source .venv/bin/activate # (Mac/Linux)
   # .venv\Scripts\activate # (Windows)

3. Install dependencies:
 pip install --upgrade pip
 pip install -r requirements.txt

 ## projectstructure

├── checkpoints/
│ ├── model_baseline.pt
│ └── model_dp.pt
├── data/
│ ├── Dataset/
│ │ ├── train.csv
│ │ ├── test.csv
│ ├── ml-100k/
│ │ ├── u.data
│ │ ├── u.genre
│ │ └── u.item
│ ├── full_dataset.csv
│ ├── preprocessing.py
│ ├── test.csv
│ └── train.csv
├── notebooks/
├── report/
├── results/
├── src/
│ ├── pycache/
│ ├── compute_dp_median.py
│ ├── data_loader.py
│ ├── evaluate.py
│ ├── model.py
│ ├── train_baseline.py
│ ├── train_dp.py
│ └── train_ibm_diffpriv.py
├── requirements.txt
└── README.md

## Example calls

### 1) Baseline without DP
python src/train_baseline.py \
  --train_csv Dataset/train.csv \
  --test_csv Dataset/test.csv \
  --output_dir checkpoints/baseline \
  --epochs 10 \
  --batch_size 256 \
  --lr 0.001 \
  --embed_dim 32 \
  --hidden_dims 64 32

### 2) Calculate DP median (optional, can also run internally in train_dp.py)
python src/compute_dp_median.py \
  --train_csv Dataset/train.csv \
  --test_csv Dataset/test.csv \
  --batch_size 1 \
  --eps_med 0.5 \
  --delta_med 1e-5 \
  --embed_dim 32 \
  --hidden_dims 64 32 \
  --public_subset 0.2
# → Result: Dataset/dp_clipping_value.txt

### 3) DP-SGD (Opacus) with hard clipping (C=1.0)
python src/train_dp.py \
  --train_csv Dataset/train.csv \
  --test_csv Dataset/test.csv \
  --output_dir checkpoints/dp_hardC \
  --epochs 10 \
  --batch_size 256 \
  --lr 0.001 \
  --noise_multiplier 1.0 \
  --max_grad_norm 1.0 \
  --delta 1e-5 \
  --embed_dim 32 \
  --hidden_dims 64 32

### 4) DP-SGD with dynamic C via DP median
python src/train_dp.py \
  --train_csv Dataset/train.csv \
  --test_csv Dataset/test.csv \
  --output_dir checkpoints/dp_dynC \
  --epochs 10 \
  --batch_size 256 \
  --lr 0.001 \
  --noise_multiplier 1.0 \
  --delta 1e-5 \
  --use_dp_median \
  --eps_med 0.5 \
  --delta_med 1e-5 \
  --public_subset 0.2 \
  --embed_dim 32 \
  --hidden_dims 64 32

### 5) IBM DiffPrivLib (DP Linear Regression)
python src/train_ibm_diffpriv.py \
  --train_csv Datensatz/train.csv \
  --test_csv Datensatz/Test.csv \
  --epsilon 1.0

### 6) Modellevaluation
python src/evaluate.py \
  --checkpoint_path checkpoints/dp_dynC/model_dp.pt \
  --train_csv Datensatz/train.csv \
  --test_csv Datensatz/test.csv \
  --batch_size 256 \
  --embed_dim 32 \
  --hidden_dims 64 32 \
  --num_workers 0

### 7) start Streamlit-App
python src/app.py

## Explanations

- `Dataset/train.csv`, `Dataset/test.csv`: Preprocessed MovieLens 100k (ratings, user ID, item ID).  
- `Dataset/dp_clipping_value.txt`: Contains a floating point value for the DP median (Clipping C).  
- `checkpoints/`: The models are saved here during training (e.g. `model_baseline.pt`, `model_dp.pt`).  
- `results_summary.csv`: Summarised metrics for all variants (ε, δ, σ, C, RMSE, MAE).  
- `privacy_vs_rmse.png`: Graphic showing the privacy utility trade-off.