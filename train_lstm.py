"""
Standalone training script for TRACE Module 2 LSTM predictor.
Run from the project root: python train_lstm.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from src.utils.config_loader import load_config
from src.data_layer.loader import (
    compute_rul,
    drop_constant_sensors,
    load_raw_data,
    run_pipeline,
)
from src.ml_layer.lstm_predictor import (
    Predictor,
    Trainer,
    build_sequences_with_unit_ids,
)

cfg = load_config()

print("\n" + "=" * 60)
print("  TRACE — Module 2: LSTM RUL Training")
print("=" * 60)

# 1. Run Module 1 pipeline (produces scaler, X_train, X_test)
pipeline = run_pipeline()

# 2. Build sequences WITH unit_ids (engine-level split requires them)
df_train, df_test, df_rul = load_raw_data()
df_train = compute_rul(df_train)
df_train = drop_constant_sensors(df_train)

window_size = cfg["data"]["window_size"]
X_tr, y_tr, unit_ids_train = build_sequences_with_unit_ids(df_train, window_size)

# Normalize using scaler saved by run_pipeline()
scaler = joblib.load(cfg["data"]["scaler_path"])
n, ws, nf = X_tr.shape
X_tr = scaler.transform(X_tr.reshape(-1, nf)).reshape(n, ws, nf).astype("float32")

# 3. Build test sequences with unit_ids
df_test = df_test.copy()
df_rul_idx = df_rul.reset_index(drop=True)
df_rul_idx["unit_id"] = df_rul_idx.index + 1

last_cycles_df = df_test.groupby("unit_id")["cycle"].max().reset_index()
last_cycles_df = last_cycles_df.merge(
    df_rul_idx.rename(columns={"rul": "RUL"}), on="unit_id"
)
df_test = df_test.merge(
    last_cycles_df[["unit_id", "cycle", "RUL"]], on=["unit_id", "cycle"], how="left"
)
df_test["RUL"] = df_test.groupby("unit_id")["RUL"].transform(lambda s: s.bfill())
for uid, grp_idx in df_test.groupby("unit_id").groups.items():
    grp = df_test.loc[grp_idx]
    if grp["RUL"].isna().any():
        last_rul   = grp["RUL"].dropna().iloc[-1]
        last_cycle = grp["cycle"].max()
        df_test.loc[grp_idx, "RUL"] = (
            last_rul + (last_cycle - grp["cycle"]).values
        )
df_test["RUL"] = df_test["RUL"].clip(upper=cfg["data"]["rul_cap"]).astype(np.float32)
df_test_clean = drop_constant_sensors(df_test)

X_te, y_te, unit_ids_test = build_sequences_with_unit_ids(df_test_clean, window_size)
n2, ws2, nf2 = X_te.shape
X_te = scaler.transform(X_te.reshape(-1, nf2)).reshape(n2, ws2, nf2).astype("float32")

# last_cycles per window (use final cycle of each unit for simplicity)
last_cycles_arr = np.zeros(len(unit_ids_test), dtype=np.int64)
for i, uid in enumerate(unit_ids_test):
    grp = df_test_clean[df_test_clean["unit_id"] == uid].sort_values("cycle")
    last_cycles_arr[i] = int(grp["cycle"].iloc[-1])

n_features = X_tr.shape[2]

# 4. Train
trainer = Trainer(cfg, n_features=n_features)
history = trainer.train(X_tr, y_tr, unit_ids_train)

# Print first 5 and last 5 epoch losses
print("\n--- Epoch Losses (first 5) ---")
for i, (tl, vl) in enumerate(
    zip(history["train_losses"][:5], history["val_losses"][:5]), 1
):
    print(f"  Epoch {i:03d}  train_loss={tl:.4f}  val_loss={vl:.4f}")

if history["epochs_trained"] > 10:
    print("  ...")
    last5_start = max(5, history["epochs_trained"] - 5)
    for i, (tl, vl) in enumerate(
        zip(
            history["train_losses"][last5_start:],
            history["val_losses"][last5_start:],
        ),
        start=last5_start + 1,
    ):
        print(f"  Epoch {i:03d}  train_loss={tl:.4f}  val_loss={vl:.4f}")

# 5. Evaluate
metrics = trainer.evaluate(X_te, y_te)
metrics["epochs_trained"] = history["epochs_trained"]
metrics["best_val_loss"]  = history["best_val_loss"]

print(f"\n--- Test Set Metrics ---")
print(f"  RMSE          : {metrics['rmse']:.4f}")
print(f"  MAE           : {metrics['mae']:.4f}")
print(f"  Epochs trained: {metrics['epochs_trained']}")

# 6. Save
trainer.save(metrics)
print(f"\n  [OK] Model saved  ->  {cfg['model']['model_path']}")
print(f"  [OK] Metadata     ->  {cfg['model']['metadata_path']}")

# 7. Fleet predictions
predictor   = Predictor(cfg, n_features=n_features)
fleet_preds = predictor.predict_fleet(X_te, unit_ids_test, last_cycles_arr)

print(f"\n--- Fleet Predictions (first 5 rows) ---")
print(fleet_preds.head(5).to_string(index=False))
print("\n" + "=" * 60)
