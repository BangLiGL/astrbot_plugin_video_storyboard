# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SAOS-3624
# Copyright 2026 BangLiGL
"""Python adaptations of upstream scan/density/readability planning (v0.6.7).

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


def motion_score(previous, current, radius=2, tau=15 / 255):
    """Count unmatched pixels after a bounded local HSV search; result is 0..1."""
    height, width = previous.shape[:2]
    matched = np.zeros((height, width), dtype=bool)
    offsets = sorted(
        ((dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)),
        key=lambda offset: abs(offset[0]) + abs(offset[1]),
    )
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        if y0 >= y1 or x0 >= x1:
            continue
        a = previous[y0:y1, x0:x1]
        b = current[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        delta = np.abs(a - b)
        hue = np.minimum(delta[..., 0], 180 - delta[..., 0]) / 180
        distance = (
            hue * np.minimum(a[..., 1], b[..., 1]) / 255 + delta[..., 1] / 255 + delta[..., 2] / 255
        ) / 3
        matched[y0:y1, x0:x1] |= distance <= tau
        if matched.all():
            return 0.0
    return float(1 - np.mean(matched))


def motion_floor(scores):
    """Upstream p75 noise calibration, converted from 0..255 to our 0..1 scale."""
    values = sorted(value * 255 for _, value in scores)
    p75 = values[min(len(values) - 1, int(len(values) * 0.75))] if values else 0
    return min(150, max(80, math.floor(80 + 14.6 * (p75 - 2.5) + 0.5))) / 255


def sparse_spikes(scores, coarse, duration):
    """Only replace sparse cuts with genuinely new, still-sparse slide changes."""
    if len(scores) <= 20 or len(coarse) / max(1, duration) >= 0.1:
        return coarse
    values = np.array([value for _, value in scores])
    baseline = float(np.median(values))
    picked = []
    for index, (at, value) in enumerate(scores):
        around = np.concatenate(
            (values[max(0, index - 12) : index], values[index + 1 : index + 13])
        )
        local = float(np.median(around)) if len(around) else baseline
        if value > max(25 / 255, 4 * max(0.2 / 255, min(local, baseline))):
            if picked and at - picked[-1][0] < 0.35:
                if value > picked[-1][1]:
                    picked[-1] = (at, value)
            else:
                picked.append((at, value))
    times = [at for at, _ in picked if 0 < at < duration]
    if len(times) <= len(coarse) or len(times) / max(duration, 1e-6) > 0.3:
        return coarse
    if len(times) >= 4:
        novel = sum(all(abs(at - other) > 2 for other in coarse) for at in times)
        if novel / len(times) < 0.4:
            return coarse
    return times


def normalize_changes(scores, means, bars=0, offset=1):
    """Keep thresholds on raw scores; compensate only comparisons for dark/letterboxed scenes."""
    content = max(0.5, 1 - bars)
    return [
        (
            at,
            value * min(3, max(1, 160 / max(20, means[min(i + offset, len(means) - 1)]))) / content,
        )
        for i, (at, value) in enumerate(scores)
    ]


def scene_cuts(
    scores, duration, normalized=None, long_scores=None, mode="abs", absolute_scores=None
):
    if not scores:
        return []
    raw = np.array([score for _, score in scores], dtype=float)
    values = np.array([score for _, score in (normalized or scores)], dtype=float)
    times = np.array([at for at, _ in scores])
    baseline = float(np.median(values))
    target = min(len(values) * 0.25, duration / max(0.5, min(1.5, duration / 60)))
    rank = max(0, min(len(values) - 1, int(len(values) - target)))
    floor = motion_floor(normalized or scores) if mode == "mc" else 0.04
    threshold = max(floor, float(np.sort(raw)[rank]))
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.04
    candidates = []
    for index, (at, _) in enumerate(scores):
        value = values[index]
        neighbours = np.concatenate(
            (values[max(0, index - 8) : index], values[index + 1 : index + 9])
        )
        local = float(np.median(neighbours)) if len(neighbours) else baseline
        spike = max(0.025, min(local, baseline) * 3)
        if (value >= threshold and value > floor) or (mode != "mc" and value > spike):
            # A cut is isolated; sustained camera motion is not a series of hard cuts.
            lo = values[np.searchsorted(times, at - 0.18) : np.searchsorted(times, at - 0.06)]
            hi = values[np.searchsorted(times, at + 0.06) : np.searchsorted(times, at + 0.18)]
            if any(
                len(side) and np.mean(side >= max(0.006, value * 0.5)) >= 0.6 for side in (lo, hi)
            ):
                continue
            if candidates and at - candidates[-1][0] <= step * 1.5:
                if value > candidates[-1][1]:
                    candidates[-1] = (at, value)
            elif 0 < at < duration:
                candidates.append((at, value))
    # Regular two-frame animation changes are not necessarily edits.
    if (
        len(candidates) > 5
        and np.median(np.diff([t for t, _ in candidates])) < step * 3.5
        and np.mean(raw) < 0.117
    ):
        candidates = []
    cuts = [float(at) for at, _ in candidates]
    if mode == "mc":
        # Sparse edits must stand above their immediate neighbours; dense edits bypass this gate.
        if len(cuts) / max(duration, 1e-6) <= 0.5:
            positions = np.searchsorted(times, cuts)
            cuts = [
                at
                for at, i in zip(cuts, positions)
                if values[i]
                > 2.5
                * max(
                    2 / 255, values[i - 1] if i else 0, values[i + 1] if i + 1 < len(values) else 0
                )
            ]
        cuts = sparse_spikes(normalized or scores, cuts, duration)
    # A short isolated peak in the 0.4s difference curve reveals a dissolve/fade.
    if long_scores:
        gradual_values = np.array([s for _, s in absolute_scores]) if absolute_scores else values
        gradual_threshold = (
            max(0.04, float(np.sort(gradual_values)[rank])) if absolute_scores else threshold
        )
        groups = []
        for at, value in long_scores:
            i = min(len(times) - 1, int(np.searchsorted(times, at)))
            if value > max(0.025, gradual_threshold * 0.65) and gradual_values[i] < max(
                0.025, gradual_threshold * 0.6
            ):
                if not groups or at - groups[-1][-1][0] > max(0.2, step * 1.5):
                    groups.append([])
                groups[-1].append((at, value))
        for group in groups:
            span = group[-1][0] - group[0][0]
            at = (group[0][0] + group[-1][0]) / 2 - 0.2
            if 0.15 <= span <= 2.5 and 0 < at < duration and all(abs(at - t) > 0.5 for t in cuts):
                cuts.append(float(at))
    return sorted(cuts)


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
        if (width + 6) * columns * ((height + 34) * rows + 24) <= budget:
            low = height
        else:
            high = height - 1
    cell_h = low
    # Even a one-pixel-high panoramic frame must fit the budget with labels.
    max_width = budget // (columns * (rows * (cell_h + 34) + 24)) - 6
    cell_w = max(1, min(int(cell_h * aspect), max_width))
    return {
        "columns": columns,
        "rows": rows,
        "cell_width": cell_w,
        "cell_height": cell_h,
        "width": columns * (cell_w + 6),
        "height": rows * (cell_h + 34) + 24,
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
