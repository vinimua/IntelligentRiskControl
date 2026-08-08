"""生成退化 Parquet 数据。用完恢复或手动删除即可。"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)

for window, label in [
    ("W0", "data.parquet.bak"),
    ("W3", "data.parquet.bak"),
]:
    df = pd.read_parquet(f"assets/data/windows/{window}/{label}")
    is_bad = df["is_bad"].values.astype(float)
    n = len(df)

    if window == "W0":
        # 正常: AUC ~0.72
        good_m, good_s, bad_m, bad_s = 0.35, 0.10, 0.65, 0.10
        df["y_pred_proba"] = np.clip(
            np.where(is_bad == 1,
                     rng.normal(bad_m, bad_s, n),
                     rng.normal(good_m, good_s, n)),
            0.001, 0.999)
    else:
        # W3: 退化 AUC ~0.55 + device_risk_score 漂移
        df["device_risk_score"] = df["device_risk_score"] * 0.5 + rng.normal(0.10, 0.18, n)
        good_m, good_s, bad_m, bad_s = 0.42, 0.18, 0.52, 0.16
        df["y_pred_proba"] = np.clip(
            np.where(is_bad == 1,
                     rng.normal(bad_m, bad_s, n),
                     rng.normal(good_m, good_s, n)),
            0.001, 0.999)

    auc = roc_auc_score(is_bad, df["y_pred_proba"])
    dev_mean = df["device_risk_score"].mean()
    df.to_parquet(f"assets/data/windows/{window}/data.parquet", index=False)
    print(f"{window}: {n} rows, AUC={auc:.3f}, device_risk_score mean={dev_mean:.3f}")

# 验证
w0 = pd.read_parquet("assets/data/windows/W0/data.parquet")
w3 = pd.read_parquet("assets/data/windows/W3/data.parquet")
print(f"delta AUC: {roc_auc_score(w0['is_bad'],w0['y_pred_proba'])-roc_auc_score(w3['is_bad'],w3['y_pred_proba']):.3f}")
