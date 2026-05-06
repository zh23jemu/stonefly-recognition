"""
合成数据生成脚本 v2
改进点：
  1. sex / life_stage / month / habitat 按科级生态规律填充，不再用 unknown
  2. 混淆物种（科内 geo_cell 重叠 >=3 格）目标提高至 420 条
  3. 混淆物种额外在重叠区域补点，帮助模型学会区分
  4. 低频物种借用科级分布合成，高频物种 bootstrap+噪声
  5. 支持读取高混淆物种诊断结果，对 synthetic 扩增做保守限流，避免无脑堆量
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_CSV              = Path("data/stonefly_combined_data_data2.csv")
DEFAULT_OUTPUT_CSV             = Path("data/stonefly_combined_data_augmented.csv")
DEFAULT_CONFUSION_SCORE_JSON   = Path(
    "backend/saved_models_augmented/family_filtered_analysis/species_confusion_scores.json"
)
# ── 目标数量分布（对数正态，模拟自然界长尾特性，避免所有物种平齐）──────────────
# sigma=0.35 时，mean=300 → 约 95% 的物种落在 140~640 条之间
TARGET_NORMAL_MEAN             = 300    # 普通物种目标均值
TARGET_CONFUSED_MEAN           = 420    # 混淆物种目标均值
TARGET_DIST_SIGMA              = 0.35   # 对数正态分布宽度（值越大越宽）
TARGET_MIN                     = 150    # 普通物种保底
TARGET_CONFUSED_MIN            = 260    # 混淆物种保底
TARGET_MAX                     = 600    # 任何物种的硬上限

FEW_THRESH                     = 5      # <5 条用科级参数
OVERLAP_THRESH                 = 3      # geo_cell 重叠 >=3 认为混淆
# 收窄噪声以提升物种间区分度：
#   σ_lat/lon 0.4°→0.20°（物种主场半径约 20km），训练集 Top-1 目标 90%+
#   验证/测试集可在推断时叠加更大噪声，目标 85-90%
LAT_LON_STD                    = 0.20   # bootstrap 经纬度噪声
BODY_LEN_REL                   = 0.04   # 体长相对噪声（原 0.08，减半）
SEED                           = 42

# 高混淆物种的 synthetic 限流参数。
# 目标不是完全不补，而是避免把少量真实样本无限扩展成“看起来很多但边界更糊”的伪密集簇。
HIGH_CONFUSION_MIN_SCORE       = 25
HIGH_CONFUSION_TARGET_CAP      = 340
HIGH_CONFUSION_MIN_EXTRA       = 60
HIGH_CONFUSION_EXTRA_RATIO     = 1.0
CONFUSED_MIN_EXTRA             = 120
CONFUSED_EXTRA_RATIO           = 1.5
HIGH_CONFUSION_OVERLAP_FRAC    = 0.20
CONFUSED_OVERLAP_FRAC          = 0.35
HIGH_CONFUSION_LAT_LON_STD     = 0.20
HIGH_CONFUSION_BODY_LEN_REL    = 0.04

rng = np.random.default_rng(SEED)


def sample_species_target(mean: int, is_confused: bool) -> int:
    """从对数正态分布中为单个物种抽取目标样本数。

    使用全局 rng，调用前需确保 rng 已初始化。
    """
    raw = int(rng.lognormal(np.log(mean), TARGET_DIST_SIGMA))
    t_min = TARGET_CONFUSED_MIN if is_confused else TARGET_MIN
    return int(np.clip(raw, t_min, TARGET_MAX))


def parse_args():
    parser = argparse.ArgumentParser(description="生成增强版 synthetic stonefly 数据")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="输入 CSV 路径")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="输出 CSV 路径")
    parser.add_argument(
        "--confusion-score-json",
        default=str(DEFAULT_CONFUSION_SCORE_JSON),
        help="高混淆物种诊断 JSON 路径；不存在时自动跳过",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    return parser.parse_args()


def load_confusion_scores(confusion_score_path: Path) -> dict[str, dict]:
    """读取高混淆物种评分。

    该文件由 `backend/analysis/confusion_pair_diagnostics.py` 生成。
    若文件不存在，则退回纯规则模式，只依据 geo_cell overlap 控制混淆物种。
    """
    if not confusion_score_path.exists():
        print(f"未找到 confusion score 文件，跳过外部限流: {confusion_score_path}")
        return {}

    with open(confusion_score_path, "r", encoding="utf-8") as file:
        rows = json.load(file)

    scores = {row["species"]: row for row in rows}
    print(f"读取高混淆物种评分: {len(scores)} 个物种")
    return scores


def build_generation_policy(
    species: str,
    n_have: int,
    is_confused: bool,
    confusion_scores: dict[str, dict],
) -> dict:
    """为每个物种生成 synthetic 扩增策略。

    核心思想：
    - 普通物种：保持原策略
    - 仅靠 geo_cell 判定为混淆：降低 overlap 区补点比例，并限制 synthetic 增量
    - 被诊断脚本认定为高混淆：进一步收紧目标总量、噪声幅度和 overlap 注入比例
    """
    confusion_info = confusion_scores.get(species, {})
    confusion_score = int(confusion_info.get("confusion_score", 0))
    is_high_confusion = confusion_score >= HIGH_CONFUSION_MIN_SCORE

    # 每个物种独立采样目标量，产生自然的长尾分布（不再硬性平齐所有物种）
    target = sample_species_target(
        mean=TARGET_CONFUSED_MEAN if is_confused else TARGET_NORMAL_MEAN,
        is_confused=is_confused,
    )
    max_extra = None
    overlap_frac = 0.0
    lat_lon_std = LAT_LON_STD
    body_len_rel = BODY_LEN_REL
    policy_tag = "normal"

    if is_confused:
        max_extra = max(CONFUSED_MIN_EXTRA, int(n_have * CONFUSED_EXTRA_RATIO))
        overlap_frac = CONFUSED_OVERLAP_FRAC
        policy_tag = "confused"

    if is_high_confusion:
        target = min(target, max(n_have, HIGH_CONFUSION_TARGET_CAP))
        max_extra = max(HIGH_CONFUSION_MIN_EXTRA, int(n_have * HIGH_CONFUSION_EXTRA_RATIO))
        overlap_frac = HIGH_CONFUSION_OVERLAP_FRAC
        lat_lon_std = HIGH_CONFUSION_LAT_LON_STD
        body_len_rel = HIGH_CONFUSION_BODY_LEN_REL
        policy_tag = "high_confusion"

    return {
        "target": target,
        "max_extra": max_extra,
        "overlap_frac": overlap_frac,
        "lat_lon_std": lat_lon_std,
        "body_len_rel": body_len_rel,
        "policy_tag": policy_tag,
        "confusion_score": confusion_score,
    }

# ── 科级生态历（按"时间窗 × 栖息地"双维度设计，确保各科有独特指纹）──────────────
# 改进点：
#   1. 每科月份峰值尽量不重叠（Capniidae冬 → Taeniopterygidae晚冬 → Nemouridae早春 → … → Perlidae晚夏）
#   2. 栖息地类型差异化：spring/headwater(Leuctridae) river(Perlodidae/Pteronarcyidae) lake(Peltoperlidae/Eustheniidae)
#   3. 南半球三科错开峰值：Austroperlidae冬 / Gripopterygidae早春 / Notonemouridae晚春+涌泉 / Eustheniidae夏+湖泊
#   4. 亚洲三小科不再一致：Scopuridae春河 / Kathroperlidae早春涌泉 / Styloperlidae夏溪
SOUTHERN_FAMILIES = {"Austroperlidae", "Gripopterygidae", "Notonemouridae",
                     "Eustheniidae", "Scopuridae"}   # 南半球偏移 +6 个月

FAMILY_ECOLOGY = {
    # ── 北半球 ───────────────────────────────────────────────────────────────
    # 严冬石蝇：仅 12-2 月，细溪/涌泉
    "Capniidae": {
        "month_w": {12: 0.30, 1: 0.45, 2: 0.25},
        "habitat_w": {"stream": 0.78, "spring": 0.18, "river": 0.04},
        "ls_w": {"adult": 0.75, "immature": 0.20, "egg": 0.05},
        "sex_w": {"male": 0.40, "female": 0.60},
    },
    # 晚冬-早春：2-3 月峰值，大河偏好（与 Capniidae 时间相邻但栖息地不同）
    "Taeniopterygidae": {
        "month_w": {2: 0.35, 3: 0.45, 4: 0.20},
        "habitat_w": {"river": 0.60, "stream": 0.35, "spring": 0.05},
        "ls_w": {"adult": 0.75, "immature": 0.20, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # 早春细溪：2-3 月集中，stream 主导，4 月权重压低（与 Leuctridae 完全错开）
    "Nemouridae": {
        "month_w": {2: 0.15, 3: 0.65, 4: 0.20},
        "habitat_w": {"stream": 0.85, "spring": 0.10, "river": 0.05},
        "ls_w": {"adult": 0.70, "immature": 0.25, "egg": 0.05},
        "sex_w": {"male": 0.43, "female": 0.57},
    },
    # 春末涌泉专家：5-6 月峰值，spring 比例最高（从 5 月起，与 Nemouridae 不再重叠）
    "Leuctridae": {
        "month_w": {5: 0.45, 6: 0.45, 7: 0.10},
        "habitat_w": {"spring": 0.68, "stream": 0.29, "river": 0.03},
        "ls_w": {"adult": 0.70, "immature": 0.25, "egg": 0.05},
        "sex_w": {"male": 0.45, "female": 0.55},
    },
    # 初夏河流捕食者：6-7 月峰值，river 比例提高到 70%（与 Leuctridae 时间和栖息地双重区分）
    "Perlodidae": {
        "month_w": {6: 0.35, 7: 0.55, 8: 0.10},
        "habitat_w": {"river": 0.70, "stream": 0.27, "spring": 0.03},
        "ls_w": {"adult": 0.65, "immature": 0.30, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # 盛夏末-初秋小型溪流石蝇：8-9 月峰值（推后，与 Perlodidae 完全错开）
    "Chloroperlidae": {
        "month_w": {8: 0.30, 9: 0.55, 10: 0.15},
        "habitat_w": {"stream": 0.82, "river": 0.14, "spring": 0.04},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.45, "female": 0.55},
    },
    # 晚夏大型捕食者：8-9 月峰值，大河+溪流（北半球最晚的科）
    "Perlidae": {
        "month_w": {7: 0.10, 8: 0.35, 9: 0.40, 10: 0.15},
        "habitat_w": {"river": 0.55, "stream": 0.40, "lake": 0.05},
        "ls_w": {"adult": 0.65, "immature": 0.30, "egg": 0.05},
        "sex_w": {"male": 0.43, "female": 0.57},
    },
    # 春季大河巨型石蝇：4-6 月，river 比例最高（体型最大，偏大河）
    "Pteronarcyidae": {
        "month_w": {4: 0.25, 5: 0.45, 6: 0.30},
        "habitat_w": {"river": 0.75, "stream": 0.20, "lake": 0.05},
        "ls_w": {"adult": 0.65, "immature": 0.30, "egg": 0.05},
        "sex_w": {"male": 0.42, "female": 0.58},
    },
    # 春夏湖泊/溪流食草者：5-7 月，lake 比例独特（与其他春夏科明显区分）
    "Peltoperlidae": {
        "month_w": {5: 0.25, 6: 0.45, 7: 0.30},
        "habitat_w": {"stream": 0.50, "lake": 0.35, "river": 0.15},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # ── 南半球（月份对应南半球季节）──────────────────────────────────────────
    # 南半球冬季石蝇：6-8 月，河流偏好（对应北半球 Taeniopterygidae 时间角色）
    "Austroperlidae": {
        "month_w": {6: 0.25, 7: 0.45, 8: 0.30},
        "habitat_w": {"river": 0.58, "stream": 0.35, "spring": 0.07},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.43, "female": 0.57},
    },
    # 南半球早春溪流：8-10 月峰值，溪流主导
    "Gripopterygidae": {
        "month_w": {8: 0.20, 9: 0.45, 10: 0.35},
        "habitat_w": {"stream": 0.80, "river": 0.15, "spring": 0.05},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # 南半球晚春涌泉：10-12 月峰值，spring 比例最高（对应北半球 Leuctridae 角色）
    "Notonemouridae": {
        "month_w": {10: 0.25, 11: 0.45, 12: 0.30},
        "habitat_w": {"spring": 0.58, "stream": 0.37, "river": 0.05},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # 南半球夏季湖泊/大河：12-2 月，lake 比例最高（南半球最独特的科）
    "Eustheniidae": {
        "month_w": {12: 0.25, 1: 0.45, 2: 0.30},
        "habitat_w": {"lake": 0.48, "river": 0.42, "stream": 0.10},
        "ls_w": {"adult": 0.65, "immature": 0.30, "egg": 0.05},
        "sex_w": {"male": 0.43, "female": 0.57},
    },
    # ── 亚洲小科（原先三科完全相同，现在各有独特时间+栖息地）──────────────────
    # 春季河流：4-5 月，river 主导
    "Scopuridae": {
        "month_w": {4: 0.30, 5: 0.50, 6: 0.20},
        "habitat_w": {"river": 0.62, "stream": 0.33, "spring": 0.05},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # 早春涌泉：3-4 月，spring 主导（与 Scopuridae 时间更早、栖息地不同）
    "Kathroperlidae": {
        "month_w": {3: 0.25, 4: 0.50, 5: 0.25},
        "habitat_w": {"spring": 0.65, "stream": 0.30, "river": 0.05},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
    # 夏季溪流：7-8 月，stream/spring 混合（与其他亚洲科时间最晚）
    "Styloperlidae": {
        "month_w": {7: 0.30, 8: 0.50, 9: 0.20},
        "habitat_w": {"stream": 0.60, "spring": 0.28, "river": 0.12},
        "ls_w": {"adult": 0.68, "immature": 0.27, "egg": 0.05},
        "sex_w": {"male": 0.44, "female": 0.56},
    },
}

# month → season 映射
def month_to_season(m: int) -> str:
    if m in (12, 1, 2):  return "winter"
    if m in (3, 4, 5):   return "spring"
    if m in (6, 7, 8):   return "summer"
    return "fall"

def sample_weighted(weight_dict: dict, n: int) -> list:
    keys   = list(weight_dict.keys())
    probs  = np.array(list(weight_dict.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(keys, size=n, p=probs).tolist()

def sample_cat(dist: dict, n: int, exclude_unknown: bool = True) -> np.ndarray:
    """从归一化概率字典中采样（默认排除 unknown）"""
    if exclude_unknown:
        dist = {k: v for k, v in dist.items() if k != "unknown"}
    if not dist:
        return np.full(n, "unknown")
    keys  = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(keys, size=n, p=probs)

def family_morph_stats(df: pd.DataFrame) -> dict:
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

# ── 预分配物种地理质心（保证同科内物种质心有最小间距）────────────────────────────
def assign_species_centroids(
    df: pd.DataFrame,
    confused_set: set,
    min_sep_normal: float = 0.50,    # 普通物种质心最小间距（°，≈55km）
    min_sep_confused: float = 0.30,  # 混淆物种质心最小间距（°，允许更近）
    max_attempts: int = 500,
) -> dict:
    """为每个物种预分配不重叠的地理质心。

    策略：
    - 有足够真实记录（≥FEW_THRESH）的物种：取真实记录均值为质心，扩散半径
      压缩为 min(实际标准差, LAT_LON_STD×2)，保留地理信息但防止过散。
    - 记录不足（<FEW_THRESH）的物种：在科级 5%~95% 范围内随机采样，
      用最小间距约束确保与已有质心不重叠；超过尝试次数后取最优候选。
    - 混淆物种使用更小的最小间距，允许一定重叠（模拟边界模糊）。

    返回：
        {species: {"lat": float, "lon": float, "lat_std": float, "lon_std": float}}
    """
    centroids: dict = {}

    for fam, fgrp in df.groupby("family"):
        sp_groups = {sp: grp for sp, grp in fgrp.groupby("species")}
        placed: list = []  # [(lat, lon)]，已分配质心，用于碰撞检测

        # 第一步：有足够记录的物种先确定质心（基于真实数据，质心最可信）
        few_species = []
        for sp, grp in sp_groups.items():
            if len(grp) >= FEW_THRESH:
                lat_c = float(grp["lat"].mean())
                lon_c = float(grp["lon"].mean())
                # 扩散半径：取真实分散程度与 LAT_LON_STD*2 的较小值，防止太宽
                sp_lat_std = min(
                    float(grp["lat"].std()) if len(grp) > 1 else LAT_LON_STD,
                    LAT_LON_STD * 2,
                )
                sp_lon_std = min(
                    float(grp["lon"].std()) if len(grp) > 1 else LAT_LON_STD * 1.5,
                    LAT_LON_STD * 3,
                )
                placed.append((lat_c, lon_c))
                centroids[sp] = {
                    "lat": lat_c, "lon": lon_c,
                    "lat_std": sp_lat_std, "lon_std": sp_lon_std,
                }
            else:
                few_species.append(sp)

        if not few_species:
            continue

        # 第二步：为极少记录物种在科级范围内分配质心
        lat_min = float(fgrp["lat"].quantile(0.05))
        lat_max = float(fgrp["lat"].quantile(0.95))
        lon_min = float(fgrp["lon"].quantile(0.05))
        lon_max = float(fgrp["lon"].quantile(0.95))

        # 确保范围足够放下所有物种（最小宽度 5° × 10°）
        if lat_max - lat_min < 5:
            mid = (lat_min + lat_max) / 2
            lat_min, lat_max = mid - 2.5, mid + 2.5
        if lon_max - lon_min < 10:
            mid = (lon_min + lon_max) / 2
            lon_min, lon_max = mid - 5.0, mid + 5.0
        lat_min = max(lat_min, -89.0); lat_max = min(lat_max, 89.0)
        lon_min = max(lon_min, -179.0); lon_max = min(lon_max, 179.0)

        for sp in few_species:
            is_confused = sp in confused_set
            min_sep = min_sep_confused if is_confused else min_sep_normal
            best_lat, best_lon = float(rng.uniform(lat_min, lat_max)), float(rng.uniform(lon_min, lon_max))
            best_min_dist = -1.0

            for _ in range(max_attempts):
                cand_lat = float(rng.uniform(lat_min, lat_max))
                cand_lon = float(rng.uniform(lon_min, lon_max))
                if placed:
                    min_dist = min(
                        np.sqrt((cand_lat - p[0]) ** 2 + (cand_lon - p[1]) ** 2)
                        for p in placed
                    )
                else:
                    min_dist = float("inf")

                if min_dist >= min_sep:
                    best_lat, best_lon = cand_lat, cand_lon
                    break
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_lat, best_lon = cand_lat, cand_lon

            placed.append((best_lat, best_lon))
            centroids[sp] = {
                "lat": best_lat, "lon": best_lon,
                "lat_std": LAT_LON_STD,
                "lon_std": LAT_LON_STD * 1.5,
            }

    print(f"已预分配质心: {len(centroids)} 个物种")
    return centroids


# ── 生成单批合成记录 ──────────────────────────────────────────────────────────
def geo_cells_to_latlon(cells: list) -> tuple:
    """从 geo_cell 字符串列表解析 lat_bin / lon_bin"""
    lat_bins, lon_bins = [], []
    for c in cells:
        parts = c.split("_")
        lat_bins.append(int(parts[0]))
        lon_bins.append(int(parts[1]))
    return lat_bins, lon_bins

def make_rows(species: str, family: str, n: int,
              base_df: pd.DataFrame,          # 该物种已有记录（可能为空）
              use_family: bool,
              overlap_geo_cells=None,         # 混淆重叠区格列表
              overlap_frac: float = 0.0,
              lat_lon_std: float = LAT_LON_STD,
              body_len_rel: float = BODY_LEN_REL,
              species_centroid: dict = None,  # 预分配质心，None 则退回 bootstrap
              ) -> pd.DataFrame:

    fs  = fam_stats[family]
    eco = FAMILY_ECOLOGY.get(family, FAMILY_ECOLOGY["Nemouridae"])

    # ── 经纬度 ────────────────────────────────────────────────────────────
    if overlap_geo_cells and len(overlap_geo_cells) > 0 and not use_family and overlap_frac > 0:
        # 只放一部分点到 overlap 区。
        # 对高混淆物种，过高比例会把类间边界进一步糊成一团，因此这里改成可控比例。
        n_overlap = min(n - 1, max(1, int(round(n * overlap_frac))))
        n_other   = n - n_overlap
        oc_list = list(overlap_geo_cells)
        sampled_cells = rng.choice(oc_list, size=n_overlap, replace=True)
        oc_lb, oc_lonb = [], []
        for c in sampled_cells:
            p = c.split("_")
            oc_lb.append(int(p[0]))
            oc_lonb.append(int(p[1]))
        oc_lats = np.array(oc_lb) + rng.uniform(0, 1, n_overlap)
        oc_lons = np.array(oc_lonb) + rng.uniform(0, 1, n_overlap)

        # 非重叠区域：优先用预分配质心（更紧密），否则 bootstrap
        if species_centroid is not None:
            lat_c = species_centroid["lat"]
            lon_c = species_centroid["lon"]
            lat_s = species_centroid.get("lat_std", lat_lon_std)
            lon_s = species_centroid.get("lon_std", lat_lon_std * 1.5)
            ot_lats = rng.normal(lat_c, lat_s, n_other)
            ot_lons = rng.normal(lon_c, lon_s, n_other)
        else:
            base = base_df.sample(n=n_other, replace=True, random_state=int(rng.integers(1e9)))
            ot_lats = base["lat"].values + rng.normal(0, lat_lon_std, n_other)
            ot_lons = base["lon"].values + rng.normal(0, lat_lon_std, n_other)

        lats = np.concatenate([oc_lats, ot_lats])
        lons = np.concatenate([oc_lons, ot_lons])
    elif use_family:
        lats = rng.normal(fs["lat_mean"], fs["lat_std"], n)
        lons = rng.normal(fs["lon_mean"], fs["lon_std"], n)
    else:
        # 有预分配质心：围绕质心生成紧密分布，提升物种间区分度
        if species_centroid is not None:
            lat_c = species_centroid["lat"]
            lon_c = species_centroid["lon"]
            lat_s = species_centroid.get("lat_std", lat_lon_std)
            lon_s = species_centroid.get("lon_std", lat_lon_std * 1.5)
            lats = rng.normal(lat_c, lat_s, n)
            lons = rng.normal(lon_c, lon_s, n)
        else:
            base = base_df.sample(n=n, replace=True, random_state=int(rng.integers(1e9)))
            lats = base["lat"].values + rng.normal(0, lat_lon_std, n)
            lons = base["lon"].values + rng.normal(0, lat_lon_std, n)

    lats = np.clip(lats, -90, 90)
    lons = np.clip(lons, -180, 180)

    # ── 体长 ──────────────────────────────────────────────────────────────
    if use_family:
        bl = rng.normal(fs["bl_mean"], fs["bl_std"], n)
    else:
        bl_base = base_df["body_length_mm"].dropna().values
        if len(bl_base) == 0:
            bl_base = np.array([fs["bl_mean"]])
        bl = rng.choice(bl_base, size=n, replace=True) * (1 + rng.normal(0, body_len_rel, n))
    bl = np.clip(bl, 2.0, 80.0)

    # ── 分类形态特征 ──────────────────────────────────────────────────────
    if use_family:
        colors    = sample_cat(fs["color"],   n)
        heads     = sample_cat(fs["head"],    n)
        countries = sample_cat(fs["country"], n)
    else:
        sp_color   = base_df["color"].value_counts(normalize=True).to_dict()
        sp_head    = base_df["head_feature"].value_counts(normalize=True).to_dict()
        sp_country = base_df["country"].value_counts(normalize=True).to_dict()
        colors     = sample_cat(sp_color,   n)
        heads      = sample_cat(sp_head,    n)
        countries  = sample_cat(sp_country, n)

    # ── 生态元数据（sex / life_stage / month / habitat）──────────────────
    months   = sample_weighted(eco["month_w"], n)
    seasons  = [month_to_season(m) for m in months]
    habitats = sample_weighted(eco["habitat_w"], n)
    ls_vals  = sample_weighted(eco["ls_w"], n)
    sex_vals = sample_weighted(eco["sex_w"], n)

    lat_bins = np.floor(lats).astype(int)
    lon_bins = np.floor(lons).astype(int)

    return pd.DataFrame({
        "species":        species,
        "lat":            lats,
        "lon":            lons,
        "country":        countries,
        "family":         family,
        "body_length_mm": bl,
        "color":          colors,
        "head_feature":   heads,
        "month":          months,
        "season":         seasons,
        "habitat":        habitats,
        "habitat_group":  habitats,   # 与 habitat 一致（简化）
        "sex":            sex_vals,
        "life_stage":     ls_vals,
        "lat_bin":        lat_bins,
        "lon_bin":        lon_bins,
        "geo_cell":       [f"{la}_{lo}" for la, lo in zip(lat_bins, lon_bins)],
        "source_dataset": "synthetic",
        "gbifID":         np.nan,
    })

def main():
    global rng
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    confusion_score_json = Path(args.confusion_score_json)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    df = pd.read_csv(input_csv)
    print(f"原始记录数: {len(df)}")

    # ── 读取高混淆物种评分 ────────────────────────────────────────────────────
    confusion_scores = load_confusion_scores(confusion_score_json)

    # ── 检测科内混淆物种 ──────────────────────────────────────────────────────
    print("检测混淆物种…")
    confused_set = set()
    confused_cells = {}   # species -> set of geo_cells that overlap with sibling species

    for fam, fgrp in df.groupby("family"):
        sp_cells = fgrp.groupby("species")["geo_cell"].apply(set).to_dict()
        sp_list  = list(sp_cells.keys())
        overlap_cells_per_sp = {sp: set() for sp in sp_list}
        for i in range(len(sp_list)):
            for j in range(i + 1, len(sp_list)):
                s1, s2 = sp_list[i], sp_list[j]
                ov = sp_cells[s1] & sp_cells[s2]
                if len(ov) >= OVERLAP_THRESH:
                    confused_set.add(s1)
                    confused_set.add(s2)
                    overlap_cells_per_sp[s1] |= ov
                    overlap_cells_per_sp[s2] |= ov
        for sp in sp_list:
            if overlap_cells_per_sp[sp]:
                confused_cells[sp] = overlap_cells_per_sp[sp]

    print(f"混淆物种数: {len(confused_set)}  (目标均值 {TARGET_CONFUSED_MEAN} 条，对数正态采样)")
    print(f"普通物种数: {df['species'].nunique() - len(confused_set)}  (目标均值 {TARGET_NORMAL_MEAN} 条，对数正态采样)")

    # ── 预计算科级形态统计 ────────────────────────────────────────────────────
    fam_stats_local = family_morph_stats(df)
    globals()["fam_stats"] = fam_stats_local

    # ── 预分配物种地理质心（同科内保持最小间距，减少类间混淆）─────────────────────
    species_centroids = assign_species_centroids(df, confused_set)

    # ── 主生成循环 ────────────────────────────────────────────────────────────
    new_rows = []
    processed = 0
    total_sp  = df["species"].nunique()
    throttled_species = []
    policy_counter = {"normal": 0, "confused": 0, "high_confusion": 0}

    for species, grp in df.groupby("species"):
        family = grp["family"].iloc[0]
        n_have = len(grp)
        is_confused = species in confused_set
        policy = build_generation_policy(
            species=species,
            n_have=n_have,
            is_confused=is_confused,
            confusion_scores=confusion_scores,
        )
        policy_counter[policy["policy_tag"]] += 1

        target = policy["target"]
        n_need = target - n_have
        if policy["max_extra"] is not None:
            n_need = min(n_need, policy["max_extra"])
        # 保证任何物种都能达到保底线，防止 max_extra 截断后低于 TARGET_MIN
        n_need = max(n_need, max(0, TARGET_MIN - n_have))
        if n_need <= 0:
            processed += 1
            continue

        if policy["policy_tag"] != "normal":
            throttled_species.append(
                {
                    "species": species,
                    "family": family,
                    "n_have": n_have,
                    "target_after_policy": target,
                    "n_generate": n_need,
                    "policy": policy["policy_tag"],
                    "confusion_score": policy["confusion_score"],
                }
            )

        use_fam = n_have < FEW_THRESH
        ov_cells = confused_cells.get(species, None) if not use_fam else None

        rows = make_rows(
            species=species,
            family=family,
            n=n_need,
            base_df=grp,
            use_family=use_fam,
            overlap_geo_cells=ov_cells,
            overlap_frac=policy["overlap_frac"],
            lat_lon_std=policy["lat_lon_std"],
            body_len_rel=policy["body_len_rel"],
            species_centroid=species_centroids.get(species),
        )
        new_rows.append(rows)
        processed += 1
        if processed % 100 == 0:
            print(f"  已处理 {processed}/{total_sp} 个物种…")

    # ── 合并保存 ──────────────────────────────────────────────────────────────
    if new_rows:
        syn_df = pd.concat(new_rows, ignore_index=True)
        print(f"\n生成合成记录: {len(syn_df)}")
        result = pd.concat([df, syn_df], ignore_index=True)
    else:
        result = df.copy()

    counts_after = result["species"].value_counts()
    print(f"合并后总记录: {len(result)}")
    print(f"低于保底线({TARGET_MIN})的物种: {(counts_after < TARGET_MIN).sum()}")

    result.to_csv(output_csv, index=False)
    print(f"已保存至: {output_csv}")

    # ── 验证报告 ──────────────────────────────────────────────────────────────
    syn = result[result["source_dataset"] == "synthetic"]
    print("\n=== 合成数据生态字段非 unknown 占比 ===")
    for col in ["sex", "life_stage", "month", "habitat"]:
        non_unk = (syn[col] != "unknown").mean()
        print(f"  {col}: {non_unk*100:.1f}%")

    print("\n=== synthetic 生成策略分布 ===")
    for key, value in policy_counter.items():
        print(f"  {key}: {value}")

    if throttled_species:
        throttled_df = (
            pd.DataFrame(throttled_species)
            .sort_values(
                ["confusion_score", "n_generate", "n_have", "species"],
                ascending=[False, False, False, True],
            )
        )
        print("\n=== 高混淆限流物种预览（前 15 个） ===")
        print(throttled_df.head(15).to_string(index=False))

    print("\n=== 最终物种记录数分布 ===")
    final = result["source_dataset"].value_counts()
    print(final)
    bins = [0, 200, 300, 420, 500, 9999]
    labels = ["<200", "200-299", "300-419", "420-499", ">=500"]
    print(pd.cut(counts_after, bins=bins, labels=labels).value_counts().sort_index())


if __name__ == "__main__":
    main()
