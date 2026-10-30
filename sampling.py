# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SAOS-3624
# Copyright 2026 BangLiGL
"""Python adaptations of upstream scan/density/readability planning (v0.5.4).

Modified for OpenCV HSV ranges, bounded AstrBot chat images, and exact canvas
geometry. See NOTICE and LICENSES/Apache-2.0.txt. No DSH token pricing is assumed.
"""

import math

import numpy as np

NEEDS = {"shot": (160, 90), "scene": (320, 180), "text": (480, 270)}


def hsv_distance(previous, current):
    """OpenCV hue is circular in [0,180); grey pixels have unreliable hue."""
    delta = np.abs(current - previous)
    hue = np.minimum(delta[..., 0], 180 - delta[..., 0]) / 180
    saturation = np.minimum(current[..., 1], previous[..., 1]) / 255
    return float(np.mean(hue * saturation + delta[..., 1] / 255 + delta[..., 2] / 255) / 3)


def scene_cuts(scores, duration):
    if not scores:
        return []
    values = np.array([score for _, score in scores], dtype=float)
    baseline = float(np.median(values))
    target = min(len(values) * 0.25, duration / max(0.5, min(1.5, duration / 60)))
    rank = max(0, min(len(values) - 1, int(len(values) - target)))
    threshold = max(0.04, float(np.sort(values)[rank]))
    candidates = []
    for index, (at, value) in enumerate(scores):
        neighbours = np.concatenate(
            (values[max(0, index - 8) : index], values[index + 1 : index + 9])
        )
        local = float(np.median(neighbours)) if len(neighbours) else baseline
        spike = max(0.025, min(local, baseline) * 3)
        if value > threshold + 1e-6 or value > spike:
            if candidates and at - candidates[-1][0] < 0.35:
                if value > candidates[-1][1]:
                    candidates[-1] = (at, value)
            elif 0 < at < duration:
                candidates.append((at, value))
    return [float(at) for at, _ in candidates]


def density_boundaries(scores, duration, count):
    """Integrate capped, smoothed change scores; quiet intervals retain coverage."""
    if count < 1 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("A positive duration and frame count are required")
    bins = min(3600, max(1, math.ceil(duration / 0.25)))
    step = duration / bins
    values, weights = np.zeros(bins), np.zeros(bins)
    usable = [(t, max(0.0, s)) for t, s in scores if 0 <= t < duration and math.isfinite(s)]
    cap = float(np.quantile([s for _, s in usable], 0.95)) if usable else 0
    for at, score in usable:
        index = min(bins - 1, int(at / step))
        values[index] += min(cap, score)
        weights[index] += 1
    present = np.flatnonzero(weights)
    if len(present):
        values[present] /= weights[present]
        values = np.interp(np.arange(bins), present, values[present])
    radius = max(1, min(bins, round(0.5 / step)))
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    smooth = np.array(
        [
            (prefix[min(bins, i + radius + 1)] - prefix[max(0, i - radius)])
            / (min(bins, i + radius + 1) - max(0, i - radius))
            for i in range(bins)
        ]
    )
    mean = float(np.mean(smooth))
    density = np.clip(smooth / mean, 1, 2.5) if mean > 0 else np.ones(bins)
    cumulative = np.concatenate(([0.0], np.cumsum(density)))
    marks = np.linspace(0, cumulative[-1], count + 1)
    boundaries = np.interp(marks, cumulative, np.linspace(0, duration, bins + 1)).tolist()
    boundaries[0], boundaries[-1] = 0.0, duration
    return boundaries


def layout(count, aspect, budget):
    """Choose compact rows/columns with a strict total image pixel ceiling."""
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("A positive finite aspect ratio is required")
    best = None
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        cost = abs(math.log(columns / rows)) + 0.25 * (rows * columns - count)
        if columns > rows:
            cost += 0.05
        if best is None or cost < best[0]:
            best = cost, columns, rows
    _, columns, rows = best
    # Each tile reserves 28 px for its timestamp and 6 px for padding.
    low, high = 1, int(math.sqrt(budget)) + 1
    while low < high:
        height = (low + high + 1) // 2
        width = max(1, int(height * aspect))
        if (width + 6) * columns * (height + 34) * rows <= budget:
            low = height
        else:
            high = height - 1
    cell_h = low
    # Even a one-pixel-high panoramic frame must fit the budget with labels.
    max_width = budget // (columns * rows * (cell_h + 34)) - 6
    cell_w = max(1, min(int(cell_h * aspect), max_width))
    return {
        "columns": columns,
        "rows": rows,
        "cell_width": cell_w,
        "cell_height": cell_h,
        "width": columns * (cell_w + 6),
        "height": rows * (cell_h + 34),
    }


def readability(width, height):
    if width < height:
        width, height = height, width
    for need in ("text", "scene", "shot"):
        target_w, target_h = NEEDS[need]
        if width >= target_w * 0.95 and height >= target_h * 0.95:
            return need
    return "composition"


def plan_images(cells, aspect, budget=1200000, need="scene", max_per_sheet=0, max_images=4):
    cells = max(1, min(32, int(cells)))
    budget = max(200000, min(2000000, int(budget)))
    need = need if need in NEEDS else "scene"
    max_images = max(1, min(6, int(max_images)))
    capacity = 1
    for count in range(1, 17):
        canvas = layout(count, aspect, budget)
        level = readability(canvas["cell_width"], canvas["cell_height"])
        if level in NEEDS and list(NEEDS).index(level) >= list(NEEDS).index(need):
            capacity = count
    if max_per_sheet:
        capacity = max(1, min(16, int(max_per_sheet)))
    count = min(cells, capacity * max_images)
    pages = math.ceil(count / capacity)
    counts = [count // pages + (index < count % pages) for index in range(pages)]
    warnings = []
    if count != cells:
        warnings.append(f"图片数量上限将采样格数从{cells}调整为{count}，仍覆盖全片，但细节可能遗漏")
    layouts = [layout(number, aspect, budget) for number in counts]
    for canvas in layouts:
        level = readability(canvas["cell_width"], canvas["cell_height"])
        if level not in NEEDS or list(NEEDS).index(level) < list(NEEDS).index(need):
            warnings.append(
                "实际格子低于所需可读性，请减少每图格数或提高像素预算；不能凭模糊画面读字"
            )
            break
    return {
        "requested_cells": cells,
        "actual_cells": count,
        "need": need,
        "pixel_budget": budget,
        "max_per_sheet": capacity,
        "counts": counts,
        "layouts": layouts,
        "warnings": warnings,
    }
