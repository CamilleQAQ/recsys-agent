"""用 Pillow 画两张实验图（环境里没有 matplotlib，Pillow 由运行时提供）。

    python make_figures.py

输出：fig1_ranker_ablation.png / fig2_stagewise_recall.png
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]
BG = (255, 255, 255)
INK = (32, 36, 44)
MUTED = (110, 118, 130)
GRID = (228, 231, 236)
ACCENT = (36, 99, 235)
NEGATIVE = (200, 60, 55)
POSITIVE = (22, 140, 90)
NEUTRAL = (150, 158, 170)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size, index=1 if bold else 0)
            except OSError:
                return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rounded_bar(draw, box, color, radius=4):
    draw.rounded_rectangle(box, radius=radius, fill=color)


def fig_ranker() -> Path:
    ablation = json.loads((ROOT / "ranker_ablation.json").read_text(encoding="utf-8"))
    recall = json.loads((ROOT / "ranker_recall_features.json").read_text(encoding="utf-8"))
    random_anchor = ablation["random_anchor"]

    arms = [
        ("现行排序器 R1", ablation["arms"]["R1"]["hit_targets"]["at20"], NEGATIVE),
        ("重训·同特征 R2", ablation["arms"]["R2"]["hit_targets"]["at20"], NEUTRAL),
        ("重训·加交互先验 R3", ablation["arms"]["R3"]["hit_targets"]["at20"], NEUTRAL),
        ("加两路召回分数 R5", recall["arms"]["R5"]["hit_targets_at20"], ACCENT),
        ("R5 + 类目重排", recall["arms"]["R5_rerank"]["hit_targets_at20"], ACCENT),
        ("融合顺序 R0", ablation["arms"]["R0"]["hit_targets"]["at20"], POSITIVE),
        ("R0 + 类目重排", ablation["arms"]["R0_rerank"]["hit_targets"]["at20"], POSITIVE),
    ]

    width, height = 1600, 980
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title = font(44, bold=True)
    subtitle = font(26)
    label_font = font(28)
    value_font = font(30, bold=True)
    small = font(22)

    draw.text((90, 60), "排序器消融：同批候选、同批用户，只换排序器", font=title, fill=INK)
    draw.text(
        (90, 120),
        "指标：Top-20 命中目标数（共 4,275 个目标 / 2,000 用户 / 每用户 100 条融合候选）",
        font=subtitle,
        fill=MUTED,
    )

    left, right = 560, 1420
    top, bottom = 220, 800
    max_value = max(row[1] for row in arms)
    scale = (right - left) / (max_value * 1.12)
    row_height = (bottom - top) / len(arms)
    bar_height = row_height * 0.52

    random_mean = random_anchor["Recall@20_mean"] * 4275
    random_low = (random_anchor["Recall@20_mean"] - random_anchor["Recall@20_std_across_permutations"]) * 4275
    random_high = (random_anchor["Recall@20_mean"] + random_anchor["Recall@20_std_across_permutations"]) * 4275

    # 随机重排区间：先画在底层，避免遮住柱子和数值
    x_low = left + random_low * scale
    x_high = left + random_high * scale
    draw.rectangle((x_low, top - 10, x_high, bottom + 10), fill=(236, 240, 248))

    for index, (label, value, color) in enumerate(arms):
        center = top + row_height * (index + 0.5)
        draw.text((left - 24, center - 18), label, font=label_font, fill=INK, anchor="ra")
        length = value * scale
        rounded_bar(draw, (left, center - bar_height / 2, left + length, center + bar_height / 2), color)
        draw.text((left + length + 18, center - 20), f"{value:.0f}", font=value_font, fill=INK)

    draw.line((left + random_mean * scale, top - 10, left + random_mean * scale, bottom + 10),
              fill=(120, 132, 152), width=3)
    draw.text((left + random_mean * scale + 12, bottom + 18),
              f"随机重排 {random_mean:.1f} ± {random_high - random_mean:.1f}", font=small, fill=MUTED)

    draw.text((90, 870),
              "随机带为同一候选池随机打乱 50 次的 ±1 个标准差；现行排序器落在随机带内且低于均值，"
              "R5 仍不及融合顺序。",
              font=small, fill=MUTED)

    out = ROOT / "fig1_ranker_ablation.png"
    image.save(out)
    return out


def fig_stagewise() -> Path:
    seed = json.loads((ROOT / "seed_study.json").read_text(encoding="utf-8"))
    deterministic = seed["deterministic"]
    aggregate = seed["aggregate"]
    arms = [
        ("Popularity", deterministic["popularity"]["metrics"]["Recall@20"], 0.0),
        ("ItemCF", deterministic["itemcf"]["metrics"]["Recall@20"], 0.0),
        ("Two-Tower", aggregate["two_tower"]["Recall@20"]["mean"], aggregate["two_tower"]["Recall@20"]["std"]),
        ("MIND", aggregate["mind"]["Recall@20"]["mean"], aggregate["mind"]["Recall@20"]["std"]),
        ("融合", aggregate["fusion"]["Recall@20"]["mean"], aggregate["fusion"]["Recall@20"]["std"]),
        ("+排序器", aggregate["ranked"]["Recall@20"]["mean"], aggregate["ranked"]["Recall@20"]["std"]),
        ("+类目重排", aggregate["final"]["Recall@20"]["mean"], aggregate["final"]["Recall@20"]["std"]),
    ]

    width, height = 1600, 980
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title = font(44, bold=True)
    subtitle = font(26)
    label_font = font(24)
    value_font = font(24, bold=True)
    small = font(22)

    draw.text((90, 60), "逐阶段 Recall@20（全部候选与测试目标固定，只换阶段）", font=title, fill=INK)
    draw.text(
        (90, 120),
        f"Two-Tower / MIND / 融合 / 排序 / 重排为 {len(seed['seeds'])} 个种子均值，误差线为 ±1 个标准差；"
        "Popularity / ItemCF 为确定性基线",
        font=subtitle, fill=MUTED,
    )

    left, right = 150, 1490
    top, bottom = 250, 780
    # 纵轴取整到 0.2 个百分点，避免刻度标签出现 0.3/0.5 这种不均匀读数
    raw_max = max(row[1] + row[2] for row in arms) * 1.25
    step_value = 0.002
    max_value = step_value * int(raw_max / step_value + 0.999)
    step = (right - left) / len(arms)
    band = step * 0.52

    ticks = int(round(max_value / step_value))
    for tick in range(ticks + 1):
        value = step_value * tick
        y = bottom - (bottom - top) * tick / ticks
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 18, y), f"{value * 100:.1f}%", font=small, fill=MUTED, anchor="rm")

    for index, (label, mean, std) in enumerate(arms):
        center = left + step * (index + 0.5)
        height_px = (mean / max_value) * (bottom - top)
        y_top = bottom - height_px
        error = (std / max_value) * (bottom - top) if std else 0.0
        color = ACCENT if index >= 2 else NEUTRAL
        rounded_bar(draw, (center - band / 2, y_top, center + band / 2, bottom), color)
        if error:
            draw.line((center, y_top - error, center, y_top + error), fill=INK, width=3)
            draw.line((center - 12, y_top - error, center + 12, y_top - error), fill=INK, width=3)
            draw.line((center - 12, y_top + error, center + 12, y_top + error), fill=INK, width=3)
        # 数值标签放在误差线上方，避免被竖线穿过
        draw.text((center, y_top - error - 34), f"{mean * 100:.2f}%", font=value_font, fill=INK, anchor="mm")
        draw.text((center, bottom + 26), label, font=label_font, fill=INK, anchor="mm")

    draw.text((90, 870),
              "排序与重排阶段使用实验二里最好的排序器（9 列物品特征 + 5 列交互先验 + 两路召回分数）。",
              font=small, fill=MUTED)

    out = ROOT / "fig2_stagewise_recall.png"
    image.save(out)
    return out


if __name__ == "__main__":
    print(fig_ranker())
    print(fig_stagewise())
