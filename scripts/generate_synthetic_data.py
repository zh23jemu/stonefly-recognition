"""
合成数据生成脚本
目标：对不足 300 条的物种补充合成记录，使每个物种至少达到 300 条。
策略：
  - 已有 >= 5 条：bootstrap + 高斯噪声
  - 已有 < 5 条：借用科级分布合成
生成记录标记为 source_dataset = "synthetic"
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
INPUT_CSV  = Path("data/stonefly_combined_data_data2.csv")
OUTPUT_CSV = Path("data/stonefly_combined_data_augmented.csv")
TARGET     = 300          # 每个物种目标条数
FEW_THRESH = 5            # 少于此值则用科级参数
SEED       = 42

# 连续特征噪声幅度（相对标准差）
LAT_LON_STD   = 0.4      # 经纬度绝对偏移（度）
BODY_LEN_REL  = 0.08     # 体长相对噪声

rng = np.random.default_rng(SEED)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"原始记录数: {len(df)}")

# ── 预计算科级统计 ─────────────────────────────────────────────────────────────
def family_stats(df: pd.DataFrame) -> dict:
    """为每个科预计算 lat/lon/body_length_mm 的均值和标准差，以及分类特征分布"""
    stats = {}
    for fam, grp in df.groupby("family"):
        bl = grp["body_length_mm"].dropna()
        stats[fam] = {
            "lat_mean":  grp["lat"].mean(),
            "lat_std":   max(grp["lat"].std(), 1.0),
            "lon_mean":  grp["lon"].mean(),
            "lon_std":   max(grp["lon"].std(), 2.0),
            "bl_mean":   bl.mean() if len(bl) > 0 else 15.0,
            "bl_std":    max(bl.std(), 1.5) if len(bl) > 0 else 3.0,
            "color":     grp["color"].value_counts(normalize=True).to_dict(),
            "head":      grp["head_feature"].value_counts(normalize=True).to_dict(),
            "country":   grp["country"].value_counts(normalize=True).to_dict(),
        }
    return stats

fam_stats = family_stats(df)

# ── 采样工具 ──────────────────────────────────────────────────────────────────
def sample_cat(dist: dict, n: int) -> np.ndarray:
    """按概率分布采样分类特征，排除 unknown"""
    keys = [k for k in dist if k != "unknown"]
    probs = np.array([dist[k] for k in keys])
    if len(keys) == 0:
        return np.full(n, "unknown")
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)

def make_geo_cell(lat_bins: np.ndarray, lon_bins: np.ndarray) -> list:
    return [f"{la}_{lo}" for la, lo in zip(lat_bins, lon_bins)]

# ── 主生成逻辑 ────────────────────────────────────────────────────────────────
new_rows = []

species_groups = df.groupby("species")
total_species  = df["species"].nunique()
processed = 0

for species, grp in species_groups:
    n_have = len(grp)
    n_need = TARGET - n_have
    if n_need <= 0:
        processed += 1
        continue

    fam = grp["family"].iloc[0]
    fs  = fam_stats[fam]

    if n_have >= FEW_THRESH:
        # ── bootstrap + 噪声 ─────────────────────────────────────
        # 有放回采样 n_need 条基础行
        base = grp.sample(n=n_need, replace=True, random_state=int(rng.integers(1e9)))

        lats = base["lat"].values + rng.normal(0, LAT_LON_STD, n_need)
        lons = base["lon"].values + rng.normal(0, LAT_LON_STD, n_need)
        lats = np.clip(lats, -90, 90)
        lons = np.clip(lons, -180, 180)

        bl_base = base["body_length_mm"].values
        missing_mask = np.isnan(bl_base)
        # 用科级均值填充缺失的体长基础值
        bl_base[missing_mask] = fs["bl_mean"]
        bl = bl_base * (1 + rng.normal(0, BODY_LEN_REL, n_need))
        bl = np.clip(bl, 2.0, 80.0)

        # 分类特征：从物种已有分布采样（排除 unknown）
        sp_color_dist = grp["color"].value_counts(normalize=True).to_dict()
        sp_head_dist  = grp["head_feature"].value_counts(normalize=True).to_dict()
        sp_country_dist = grp["country"].value_counts(normalize=True).to_dict()

        colors   = sample_cat(sp_color_dist, n_need)
        heads    = sample_cat(sp_head_dist,  n_need)
        countries = sample_cat(sp_country_dist, n_need)

    else:
        # ── 科级分布合成 ─────────────────────────────────────────
        lats = rng.normal(fs["lat_mean"], fs["lat_std"], n_need)
        lons = rng.normal(fs["lon_mean"], fs["lon_std"], n_need)
        lats = np.clip(lats, -90, 90)
        lons = np.clip(lons, -180, 180)

        bl = rng.normal(fs["bl_mean"], fs["bl_std"], n_need)
        bl = np.clip(bl, 2.0, 80.0)

        colors    = sample_cat(fs["color"],   n_need)
        heads     = sample_cat(fs["head"],    n_need)
        countries = sample_cat(fs["country"], n_need)

    lat_bins = np.floor(lats).astype(int)
    lon_bins = np.floor(lons).astype(int)

    rows = pd.DataFrame({
        "species":       species,
        "lat":           lats,
        "lon":           lons,
        "country":       countries,
        "family":        fam,
        "body_length_mm": bl,
        "color":         colors,
        "head_feature":  heads,
        "month":         "unknown",
        "season":        "unknown",
        "habitat":       "unknown",
        "habitat_group": "unknown",
        "sex":           "unknown",
        "life_stage":    "unknown",
        "lat_bin":       lat_bins,
        "lon_bin":       lon_bins,
        "geo_cell":      make_geo_cell(lat_bins, lon_bins),
        "source_dataset": "synthetic",
        "gbifID":        np.nan,
    })
    new_rows.append(rows)
    processed += 1
    if processed % 100 == 0:
        print(f"  已处理 {processed}/{total_species} 个物种…")

# ── 合并保存 ──────────────────────────────────────────────────────────────────
if new_rows:
    synthetic_df = pd.concat(new_rows, ignore_index=True)
    print(f"\n生成合成记录: {len(synthetic_df)}")
    result = pd.concat([df, synthetic_df], ignore_index=True)
else:
    result = df.copy()

# 验证各物种数量
counts_after = result["species"].value_counts()
below_target = (counts_after < TARGET).sum()
print(f"合并后总记录: {len(result)}")
print(f"仍低于 {TARGET} 条的物种: {below_target}")

result.to_csv(OUTPUT_CSV, index=False)
print(f"\n已保存至: {OUTPUT_CSV}")

# ── 简要统计报告 ──────────────────────────────────────────────────────────────
print("\n=== 最终物种记录数分布 ===")
final_counts = result["species"].value_counts()
print(f"< 300 条: {(final_counts < 300).sum()} 个物种")
print(f"300-499 条: {((final_counts >= 300) & (final_counts < 500)).sum()} 个物种")
print(f">= 500 条: {(final_counts >= 500).sum()} 个物种")
print(f"\nsource_dataset 分布:")
print(result["source_dataset"].value_counts())
