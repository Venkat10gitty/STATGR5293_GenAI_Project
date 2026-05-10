"""
MLP Fusion Model for Emotional Incongruence Detection

24-dimensional input:
  [0:7]   face_vector   — DeepFace 7-class emotion probabilities
  [7:14]  voice_vector  — wav2vec2 7-class (3 zeros for missing classes)
  [14:21] text_vector   — RoBERTa 7-class emotion probabilities
  [21]    mas_face_voice — cosine similarity (face↔voice)
  [22]    mas_face_text  — cosine similarity (face↔text)
  [23]    mas_voice_text — cosine similarity (voice↔text)

Test results (MELD, 2,615 clips):
  F1=0.8997  Precision=0.8733  Recall=0.9276  AUROC=0.9291

Usage:
    python models/mlp_fusion.py --train \
        --labels_csv data/sample_labels.csv \
        --output_dir models
"""

import torch, torch.nn as nn, numpy as np, pandas as pd
import ast, pickle, json, argparse, os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, roc_auc_score,
                              classification_report)


def parse_vector(val) -> np.ndarray:
    """Parse stored emotion vector string → numpy array."""
    try:
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        return np.array(ast.literal_eval(str(val)), dtype=np.float32)
    except Exception:
        return np.ones(7, dtype=np.float32) / 7.0


class IncongruenceMLP(nn.Module):
    """
    3-layer MLP for binary incongruence classification.

    Architecture:
        Linear(24→128) → BatchNorm → ReLU → Dropout(0.3)
        Linear(128→64) → BatchNorm → ReLU → Dropout(0.3)
        Linear(64→1)   → Sigmoid

    Trained with AdamW + CosineAnnealingLR + BCEWithLogitsLoss.
    Early stopping at epoch 39 (patience=10).
    """
    def __init__(self, input_dim=24, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_feature_matrix(df: pd.DataFrame):
    """Build 24d feature matrix from labels DataFrame."""
    X, y, splits = [], [], []
    for _, row in df.iterrows():
        features = np.concatenate([
            parse_vector(row["face_vector"]),
            parse_vector(row["voice_vector"]),
            parse_vector(row["text_vector"]),
            [row["mas_face_voice"],
             row["mas_face_text"],
             row["mas_voice_text"]],
        ]).astype(np.float32)
        X.append(features)
        y.append(int(row["incongruent"]))
        splits.append(row["split"])
    return np.array(X), np.array(y), np.array(splits)


def train(labels_csv: str, output_dir: str, epochs: int = 100):
    """Train MLP with early stopping. Saves weights + scaler."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(labels_csv)
    X, y, splits = build_feature_matrix(df)

    X_tr=X[splits=="train"]; y_tr=y[splits=="train"]
    X_v =X[splits=="dev"];   y_v =y[splits=="dev"]
    X_te=X[splits=="test"];  y_te=y[splits=="test"]

    sc = StandardScaler()
    X_tr=sc.fit_transform(X_tr); X_v=sc.transform(X_v); X_te=sc.transform(X_te)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = IncongruenceMLP().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    crit   = nn.BCEWithLogitsLoss()
    dl     = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    best_f1, best_state, patience, pc = 0.0, None, 10, 0

    print(f"{'Epoch':>6} {'Loss':>10} {'ValF1':>8}")
    for epoch in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = (torch.sigmoid(model(
                torch.tensor(X_v, dtype=torch.float32).to(device)))
                .cpu().numpy() > 0.5).astype(int)
        vf1 = f1_score(y_v, vp, zero_division=0)

        if epoch % 10 == 0:
            print(f"{epoch:>6} {loss_sum/len(dl):>10.4f} {vf1:>8.4f}")

        if vf1 > best_f1:
            best_f1   = vf1
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(
            torch.tensor(X_te, dtype=torch.float32).to(device))).cpu().numpy()
    preds = (tp > 0.5).astype(int)
    f1    = f1_score(y_te, preds, zero_division=0)

    print(f"\nTest F1: {f1:.4f}")
    print(classification_report(y_te, preds,
          target_names=["Congruent","Incongruent"]))

    torch.save(best_state, f"{output_dir}/mlp_fusion.pt")
    with open(f"{output_dir}/scaler.pkl","wb") as f:
        pickle.dump(sc, f)
    with open(f"{output_dir}/mlp_results.json","w") as f:
        json.dump({"f1": round(f1,4),
                   "precision": round(precision_score(y_te,preds,zero_division=0),4),
                   "recall":    round(recall_score(y_te,preds,zero_division=0),4),
                   "auroc":     round(roc_auc_score(y_te,tp),4)}, f, indent=2)
    print(f"✓ Saved to {output_dir}/")


def predict(features: np.ndarray,
            model_path: str = "models/mlp_fusion.pt",
            scaler_path: str = "models/scaler.pkl") -> tuple:
    """
    Inference on a single 24-dimensional feature vector.

    Args:
        features:    numpy array of shape (24,)
        model_path:  path to mlp_fusion.pt
        scaler_path: path to scaler.pkl

    Returns:
        tuple: (is_incongruent: bool, confidence: float)

    Example:
        >>> import numpy as np
        >>> features = np.random.rand(24).astype(np.float32)
        >>> is_inc, conf = predict(features)
        >>> print(f"Incongruent: {is_inc}, confidence: {conf}")
    """
    with open(scaler_path,"rb") as f:
        sc = pickle.load(f)
    model = IncongruenceMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    x = torch.tensor(sc.transform([features]), dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob > 0.5, round(prob, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--labels_csv",  default="data/sample_labels.csv")
    parser.add_argument("--output_dir",  default="models")
    parser.add_argument("--epochs",      type=int, default=100)
    args = parser.parse_args()
    if args.train:
        train(args.labels_csv, args.output_dir, args.epochs)
