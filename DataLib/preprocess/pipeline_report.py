#!/usr/bin/env python3
"""Generate one Markdown report for preprocess, wash, retarget, and QA."""
from __future__ import annotations

import argparse
import json
import pathlib
import time


BINS = [
    (0, 1, "<1"), (1, 2, "1-2"), (2, 3, "2-3"), (3, 5, "3-5"),
    (5, 10, "5-10"), (10, 20, "10-20"), (20, 30, "20-30"),
    (30, 60, "30-60"), (60, 120, "60-120"),
    (120, 300, "120-300"), (300, 600, "300-600"),
    (600, 1800, "600-1800"), (1800, float("inf"), ">1800"),
]


def load(path: str | None) -> dict:
    if not path:
        return {}
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def change(after: float, before: float, unit: str = "") -> str:
    delta = after - before
    pct = 100.0 * delta / before if before else 0.0
    return f"{delta:+,.2f}{unit} ({pct:+.1f}%)"


def threshold(counts: dict, value: float) -> int:
    for key, count in counts.items():
        if abs(float(key) - value) < 1e-9:
            return int(count)
    return 0


def duration_section(title: str, data: dict) -> list[str]:
    histogram = {
        str(item["label"]): int(item["count"])
        for item in data.get("histogram", [])
    }
    values = [
        float(value)
        for rows in data.get("per_cat", {}).values()
        for value in rows
    ]
    n = int(data.get("n", len(values)))
    total_h = float(data.get("total_h", sum(values) / 3600.0))
    lines = [
        f"### {title}",
        "",
        f"n={n:,}, total={total_h:.2f} h",
        "",
        "| bin (s) | segments | % |",
        "|---|---:|---:|",
    ]
    for lo, hi, label in BINS:
        count = histogram.get(
            label,
            sum(1 for value in values if lo <= value < hi),
        )
        pct = 100.0 * count / n if n else 0.0
        lines.append(f"| {label} | {count:,} | {pct:.1f}% |")
    p = data.get("percentiles", {})
    if p:
        get = lambda q: float(p.get(str(q), p.get(q, 0.0)))
        lines.extend([
            "",
            (
                f"Percentiles: p1={get(1):.2f}s, p5={get(5):.2f}s, "
                f"p10={get(10):.2f}s, p25={get(25):.2f}s, "
                f"**p50={get(50):.2f}s**, p75={get(75):.2f}s, "
                f"p90={get(90):.2f}s, p95={get(95):.2f}s, p99={get(99):.2f}s; "
                f"min={float(data.get('min', 0)):.2f}s, "
                f"max={float(data.get('max', 0)):.2f}s, "
                f"mean={float(data.get('mean', 0)):.2f}s."
            ),
        ])
    lines.append("")
    return lines


def wash_section(wash: dict, cleaned: dict, clean_duration: dict) -> list[str]:
    g = wash.get("global", {})
    params = wash.get("parameters", {})
    input_frames = int(g.get("input_frames", 0))
    output_frames = int(g.get("output_frames", 0))
    input_segments = int(g.get("input_files", 0))
    output_segments = int(g.get("kept_segments", 0))
    input_duration_s = float(cleaned.get("total_dur_h", 0.0)) * 3600.0
    output_duration_s = float(g.get("output_duration_s", 0.0))
    overlap = int(g.get("overlap_frames", 0))
    crouch_only = int(g.get("crouch_frames", 0)) - overlap
    steering_only = int(g.get("steering_frames", 0)) - overlap

    lines = [
        "## Stage B — Crouch/steering wash",
        "",
        "Rules:",
        "",
        "- crouch: `cmd.crouch == true`",
        (
            f"- steering: `abs(cmd.move[0]) > {float(params.get('eps_side', 0.25))}` "
            f"and `hypot(cmd.move) > {float(params.get('eps_mag', 0.2))}`"
        ),
        "- `cmd.jump`, `desYR`, `dLookY`, and look yaw are not filtered.",
        "",
        "### Cleaning stats",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| input files (preprocessed segments) | {input_segments:,} |",
        f"| input frames | {input_frames:,} |",
        f"| candidate good runs | {int(g.get('candidate_runs', 0)):,} |",
        f"| **output segments** | **{output_segments:,}** |",
        (
            f"| **output frames** | **{output_frames:,}** "
            f"({100*output_frames/input_frames:.1f}% retained) |"
            if input_frames else "| **output frames** | **0** |"
        ),
        f"| crouch frames | {int(g.get('crouch_frames', 0)):,} |",
        f"| steering frames | {int(g.get('steering_frames', 0)):,} |",
        f"| crouch ∩ steering frames | {overlap:,} |",
        f"| bad union frames removed | {int(g.get('bad_union_frames', 0)):,} |",
        (
            f"| short good runs rejected | {int(g.get('rejected_segments', 0)):,} "
            f"({int(g.get('rejected_good_frames', 0)):,} frames) |"
        ),
        f"| files without crouch/steering | {int(g.get('files_without_bad', 0)):,} |",
        f"| files with no retained output | {int(g.get('files_all_rejected', 0)):,} |",
        "",
        "### Before vs after",
        "",
        "| metric | before | after | change |",
        "|---|---:|---:|---:|",
        (
            f"| segments | {input_segments:,} | **{output_segments:,}** | "
            f"{output_segments-input_segments:+,} |"
        ),
        (
            f"| total frames | {input_frames:,} | **{output_frames:,}** | "
            f"{output_frames-input_frames:+,} "
            f"({100*(output_frames-input_frames)/input_frames:+.1f}%) |"
            if input_frames else "| total frames | 0 | **0** | 0 |"
        ),
        (
            f"| real segment-span duration | {input_duration_s/3600:.2f} h | "
            f"**{output_duration_s/3600:.2f} h** | "
            f"{(output_duration_s-input_duration_s)/3600:+.2f} h |"
        ),
        "",
        "### Bad-frame composition",
        "",
        "| type | frames | % of input |",
        "|---|---:|---:|",
    ]
    for label, count in (
        ("crouch only", crouch_only),
        ("steering only", steering_only),
        ("crouch and steering", overlap),
        ("**union removed by mask**", int(g.get("bad_union_frames", 0))),
    ):
        pct = 100.0 * count / input_frames if input_frames else 0.0
        lines.append(f"| {label} | {count:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "### Per-category distribution after wash",
        "",
        "| category | input segs | output segs | input frames | output frames | retained | output hours |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for category, row in wash.get("per_category", {}).items():
        before = int(row.get("input_frames", 0))
        after = int(row.get("output_frames", 0))
        retained = 100.0 * after / before if before else 0.0
        lines.append(
            f"| {category} | {int(row.get('input_files', 0)):,} | "
            f"{int(row.get('kept_segments', 0)):,} | {before:,} | {after:,} | "
            f"{retained:.1f}% | {float(row.get('output_duration_s', 0))/3600:.2f} |"
        )
    retained = 100.0 * output_frames / input_frames if input_frames else 0.0
    lines.extend([
        (
            f"| **total** | **{input_segments:,}** | **{output_segments:,}** | "
            f"**{input_frames:,}** | **{output_frames:,}** | "
            f"**{retained:.1f}%** | **{output_duration_s/3600:.2f}** |"
        ),
        "",
    ])
    lines.extend(duration_section("Input segment duration distribution", clean_duration))
    lines.extend(duration_section(
        "Output segment duration distribution",
        wash.get("output_duration_distribution", {}),
    ))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-stats", required=True)
    ap.add_argument("--raw-eval", required=True)
    ap.add_argument("--clean-eval", required=True)
    ap.add_argument(
        "--clean-duration",
        default=None,
        help=(
            "optional seg_duration_dist JSON; when omitted, use the "
            "pre-wash duration distribution embedded in wash_stats.json"
        ),
    )
    ap.add_argument("--wash-stats", required=True)
    ap.add_argument("--wash-report", default=None,
                    help="legacy standalone report path (not inlined)")
    ap.add_argument("--retarget-stats", required=True)
    ap.add_argument("--qa-stats", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    clean_stats = load(args.clean_stats)
    raw = load(args.raw_eval)
    cleaned = load(args.clean_eval)
    clean_duration = load(args.clean_duration)
    wash = load(args.wash_stats)
    if not clean_duration:
        clean_duration = wash.get("input_duration_distribution", {})
    retarget = load(args.retarget_stats)
    qa = load(args.qa_stats)

    raw_frames = int(raw.get("total_frames", 0))
    clean_frames = int(cleaned.get("total_frames", 0))
    raw_hours = float(raw.get("total_dur_h", 0.0))
    clean_hours = float(cleaned.get("total_dur_h", 0.0))
    lines = [
        "# MorphData_v1 preprocessing → crouch/steering wash → G1 report",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "",
        "## Retained stage directories",
        "",
        "| stage | directory | mutation policy |",
        "|---|---|---|",
        f"| Stage 0 — raw | `{pathlib.Path(args.raw_root).resolve()}` | read-only, retained |",
        f"| Stage 1 — preprocessed | `{wash.get('data_root', '')}` | new directory, retained |",
        f"| Stage 2 — crouch/steering | `{wash.get('out_root', '')}` | new directory, retained |",
        f"| Stage 3 — G1 | `{retarget.get('output_root', '')}` | new directory, retained |",
        "",
        "## Stage A — Existing MorphData preprocess",
        "",
        "### Cleaning stats",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| input files (segments) | {int(raw.get('total_files', 0)):,} |",
        f"| input frames | {raw_frames:,} |",
        f"| **output segments** | **{int(cleaned.get('total_files', 0)):,}** |",
        (
            f"| **output frames** | **{clean_frames:,}** "
            f"({100*clean_frames/raw_frames:.1f}% retained) |"
            if raw_frames else "| **output frames** | **0** |"
        ),
        f"| dropped stuck frames (dt<0.005) | {int(clean_stats.get('stuck', 0)):,} |",
        f"| deleted glitch frames | {int(clean_stats.get('glitch', 0)):,} |",
        (
            f"| split points (big gaps + teleports) | "
            f"{int(clean_stats.get('splits', 0)):,} |"
        ),
        f"| small-gap interpolated (0.05<dt<=0.1) | {int(clean_stats.get('gaps', 0)):,} |",
        f"| dropped segments (<60 frames or <1s) | {int(clean_stats.get('segs_dropped', 0)):,} |",
        f"| unchanged files | {int(clean_stats.get('files_unchanged', 0)):,} |",
        "",
        "### Before vs after (disp/dt real displacement speed)",
        "",
        "| metric | before | after | change |",
        "|---|---:|---:|---:|",
        (
            f"| segments | {int(raw.get('total_files', 0)):,} | "
            f"**{int(cleaned.get('total_files', 0)):,}** | "
            f"{int(cleaned.get('total_files', 0))-int(raw.get('total_files', 0)):+,} |"
        ),
        (
            f"| total frames | {raw_frames:,} | **{clean_frames:,}** | "
            f"{clean_frames-raw_frames:+,} "
            f"({100*(clean_frames-raw_frames)/raw_frames:+.1f}%) |"
            if raw_frames else "| total frames | 0 | **0** | 0 |"
        ),
        (
            f"| total duration | {raw_hours:.2f} h | **{clean_hours:.2f} h** | "
            f"{change(clean_hours, raw_hours, ' h')} |"
        ),
        f"| stuck frames (dt<0.005) | {int(raw.get('n_stuck', 0)):,} | **{int(cleaned.get('n_stuck', 0)):,}** | — |",
        f"| big gaps (dt>0.1) | {int(raw.get('n_gap_big', 0)):,} | **{int(cleaned.get('n_gap_big', 0)):,}** | — |",
        f"| global max dt | {float(raw.get('max_dt_global', 0)):.3f} s | **{float(cleaned.get('max_dt_global', 0)):.3f} s** | — |",
        f"| per-segment median dt | {float(raw.get('median_of_med_dt', 0)):.5f} s | **{float(cleaned.get('median_of_med_dt', 0)):.5f} s** | — |",
        f"| disp/dt speed peak | {float(raw.get('global_max_sp', 0)):.1f} m/s | **{float(cleaned.get('global_max_sp', 0)):.1f} m/s** | — |",
        "",
        "### Threshold hits after preprocess",
        "",
        "| speed threshold | frames hit |",
        "|---|---:|",
    ]
    for th in (6, 8, 10, 15, 20, 30):
        lines.append(
            f"| > {th} m/s | {threshold(cleaned.get('threshold_counts', {}), th):,} |"
        )
    lines.extend([
        "",
        "### Per-category distribution after preprocess",
        "",
        "| category | segments | frames | hours | disp/dt peak (m/s) |",
        "|---|---:|---:|---:|---:|",
    ])
    for category, row in cleaned.get("per_cat", {}).items():
        lines.append(
            f"| {category} | {int(row['files']):,} | {int(row['frames']):,} | "
            f"{float(row['dur'])/3600:.2f} | {float(row['max_sp']):.1f} |"
        )
    lines.append(
        f"| **total** | **{int(cleaned.get('total_files', 0)):,}** | "
        f"**{clean_frames:,}** | **{clean_hours:.2f}** | "
        f"**{float(cleaned.get('global_max_sp', 0)):.1f}** |"
    )
    lines.append("")
    lines.extend(duration_section("Duration distribution after preprocess", clean_duration))

    lines.extend(["---", ""])
    lines.extend(wash_section(wash, cleaned, clean_duration))

    lines.extend([
        "---",
        "",
        "## Stage C — G1 retarget and QA",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| wash input segments | {int(retarget.get('input_segments', 0)):,} |",
        f"| **retarget success segments** | **{int(retarget.get('success_segments', 0)):,}** |",
        f"| retarget rejects | {int(retarget.get('retarget_rejects', 0)):,} |",
        f"| qpos frames | {int(retarget.get('success_frames', 0)):,} |",
        f"| qpos duration | {float(retarget.get('success_duration_h', 0)):.2f} h |",
        f"| wall time | {float(retarget.get('elapsed_s', 0))/3600:.2f} h |",
        f"| QA status | **{qa.get('status', 'not-run')}** |",
        f"| QA failures | {int(qa.get('failures', 0)):,} |",
        f"| unique segment IDs checked | {int(qa.get('unique_seg_ids', 0)):,} |",
        f"| qpos frames checked | {int(qa.get('qpos_frames_checked', 0)):,} |",
        f"| washed source frames re-scanned | {int(qa.get('source_frames_checked', 0)):,} |",
        "",
        "### Per-category G1 output",
        "",
        "| category | segments | frames | hours |",
        "|---|---:|---:|---:|",
    ])
    for category, row in retarget.get("per_category", {}).items():
        lines.append(
            f"| {category} | {int(row['segments']):,} | {int(row['frames']):,} | "
            f"{float(row['duration_s'])/3600:.2f} |"
        )
    lines.append("")

    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp.replace(output)
    print(output)


if __name__ == "__main__":
    main()
