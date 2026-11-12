#!/usr/bin/env python3
"""Midpoint acoustic extraction with automatic Burg parameter auditing."""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
import tgt


APP_TITLE = "Midpoint Acoustic Extraction + Formant Audit"
FORMANT_CEILINGS_HZ = tuple(range(4000, 5501, 250))
FORMANT_COUNTS = (4.0, 4.5, 5.0, 5.5)
JUMP_RATIO_LIMIT = 1.20
BANDWIDTH_RATIO_LIMIT = 0.30
F2_F3_CLOSE_HZ = 300.0


def read_textgrid(path: Path) -> tgt.core.TextGrid:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return tgt.read_textgrid(str(path), encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not read TextGrid. " + " | ".join(errors))


def get_interval_tier(textgrid: tgt.core.TextGrid, tier_name: str) -> Any:
    wanted = tier_name.strip().casefold()
    for tier in textgrid.tiers:
        if tier.name.strip().casefold() == wanted:
            if not hasattr(tier, "intervals"):
                raise ValueError(f"Tier '{tier.name}' is not an interval tier.")
            return tier
    available = ", ".join(tier.name for tier in textgrid.tiers)
    raise ValueError(
        f"Tier '{tier_name}' not found. Available tiers: {available or '(none)'}"
    )


def optional_float(value: Any, *, positive: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if positive and number <= 0:
        return None
    return number


def rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def build_upload_map(files: list[Any]) -> tuple[dict[str, Any], list[str]]:
    mapping: dict[str, Any] = {}
    duplicates: list[str] = []
    for uploaded in files:
        key = Path(uploaded.name).stem.casefold()
        if key in mapping:
            duplicates.append(Path(uploaded.name).stem)
        else:
            mapping[key] = uploaded
    return mapping, duplicates


def collect_qc_times(
    tier: Any,
    sound_duration_s: float,
    step_s: float,
    edge_trim_s: float,
    skip_empty: bool,
) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    interval_ids: list[int] = []

    for interval_index, interval in enumerate(tier.intervals, start=1):
        label = interval.text.strip()
        if skip_empty and not label:
            continue

        start_s = max(0.0, float(interval.start_time))
        end_s = min(sound_duration_s, float(interval.end_time))
        duration_s = end_s - start_s
        if duration_s <= 0:
            continue

        trim_s = edge_trim_s if duration_s > 2 * edge_trim_s + step_s else 0.0
        frame_times = np.arange(
            start_s + trim_s,
            end_s - trim_s + step_s * 0.25,
            step_s,
            dtype=float,
        )
        if frame_times.size == 0:
            frame_times = np.asarray([(start_s + end_s) / 2.0])
        times.extend(frame_times.tolist())
        interval_ids.extend([interval_index] * len(frame_times))

    if not times:
        raise ValueError("No analyzable intervals remain for the formant audit.")
    return np.asarray(times), np.asarray(interval_ids)


def candidate_tracks(
    sound: parselmouth.Sound,
    times_s: np.ndarray,
    maximum_formant_hz: float,
    number_of_formants: float,
    time_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    formant = sound.to_formant_burg(
        time_step=time_step_s,
        max_number_of_formants=number_of_formants,
        maximum_formant=maximum_formant_hz,
    )
    values = np.full((len(times_s), 3), np.nan, dtype=np.float32)
    bandwidths = np.full((len(times_s), 3), np.nan, dtype=np.float32)

    for row_index, time_s in enumerate(times_s):
        for formant_index in range(1, 4):
            value = optional_float(
                formant.get_value_at_time(formant_index, float(time_s)),
                positive=True,
            )
            bandwidth = optional_float(
                formant.get_bandwidth_at_time(formant_index, float(time_s)),
                positive=True,
            )
            if value is not None:
                values[row_index, formant_index - 1] = value
            if bandwidth is not None:
                bandwidths[row_index, formant_index - 1] = bandwidth
    return values, bandwidths


def calculate_metrics(
    values: np.ndarray,
    bandwidths: np.ndarray,
    interval_ids: np.ndarray,
) -> dict[str, float]:
    finite = np.isfinite(values)
    coverage = np.mean(finite, axis=0)
    all_valid = np.all(finite, axis=1)

    if np.any(all_valid):
        ordered = (
            (values[:, 0] < values[:, 1])
            & (values[:, 1] < values[:, 2])
            & all_valid
        )
        ordering_error = 1.0 - float(np.sum(ordered) / np.sum(all_valid))
        convergence_rate = float(
            np.mean((values[all_valid, 2] - values[all_valid, 1]) < F2_F3_CLOSE_HZ)
        )
    else:
        ordering_error = 1.0
        convergence_rate = math.nan

    jump_rates: list[float] = []
    for column in range(3):
        series = values[:, column].astype(float)
        valid_pairs = (
            np.isfinite(series[:-1])
            & np.isfinite(series[1:])
            & (interval_ids[:-1] == interval_ids[1:])
        )
        if np.any(valid_pairs):
            ratios = np.maximum(
                series[1:][valid_pairs] / series[:-1][valid_pairs],
                series[:-1][valid_pairs] / series[1:][valid_pairs],
            )
            jump_rates.append(float(np.mean(ratios > JUMP_RATIO_LIMIT)))
        else:
            jump_rates.append(1.0)

    bandwidth_valid = finite & np.isfinite(bandwidths)
    if np.any(bandwidth_valid):
        ratios = np.divide(
            bandwidths,
            values,
            out=np.full_like(values, np.nan, dtype=float),
            where=bandwidth_valid,
        )
        bandwidth_penalty = float(
            np.mean(ratios[bandwidth_valid] > BANDWIDTH_RATIO_LIMIT)
        )
    else:
        bandwidth_penalty = 1.0

    missing_rate = 1.0 - float(np.mean(coverage))
    mean_jump_rate = float(np.mean(jump_rates))
    base_score = (
        2.0 * missing_rate
        + 3.0 * ordering_error
        + 2.0 * mean_jump_rate
        + bandwidth_penalty
    )
    return {
        "n_frames": float(len(values)),
        "F1_coverage": float(coverage[0]),
        "F2_coverage": float(coverage[1]),
        "F3_coverage": float(coverage[2]),
        "mean_coverage": float(np.mean(coverage)),
        "ordering_error_rate": ordering_error,
        "F1_jump_rate": jump_rates[0],
        "F2_jump_rate": jump_rates[1],
        "F3_jump_rate": jump_rates[2],
        "mean_jump_rate": mean_jump_rate,
        "bandwidth_penalty_rate": bandwidth_penalty,
        "F2_F3_close_rate": convergence_rate,
        "base_score": base_score,
    }


def neighboring_candidates(
    ceiling_hz: int,
    number_of_formants: float,
) -> list[tuple[int, float]]:
    ceiling_index = FORMANT_CEILINGS_HZ.index(ceiling_hz)
    count_index = FORMANT_COUNTS.index(number_of_formants)
    neighbors: list[tuple[int, float]] = []
    for offset in (-1, 1):
        adjacent_ceiling = ceiling_index + offset
        if 0 <= adjacent_ceiling < len(FORMANT_CEILINGS_HZ):
            neighbors.append(
                (FORMANT_CEILINGS_HZ[adjacent_ceiling], number_of_formants)
            )
        adjacent_count = count_index + offset
        if 0 <= adjacent_count < len(FORMANT_COUNTS):
            neighbors.append((ceiling_hz, FORMANT_COUNTS[adjacent_count]))
    return neighbors


def trajectory_distance(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > 0)
    if not np.any(valid):
        return 1.0
    return float(np.median(np.abs(np.log(first[valid] / second[valid]))))


def recording_grade(row: pd.Series) -> str:
    if (
        row["mean_coverage"] >= 0.90
        and row["ordering_error_rate"] <= 0.01
        and row["mean_jump_rate"] <= 0.03
        and row["bandwidth_penalty_rate"] <= 0.15
        and row["parameter_sensitivity"] <= 0.05
    ):
        return "A — stable"
    if (
        row["mean_coverage"] >= 0.75
        and row["ordering_error_rate"] <= 0.05
        and row["mean_jump_rate"] <= 0.10
        and row["parameter_sensitivity"] <= 0.12
    ):
        return "B — inspect"
    return "C — unreliable"


def audit_formant_parameters(
    audio_path: Path,
    textgrid_path: Path,
    tier_name: str,
    skip_empty: bool,
    qc_step_s: float,
    edge_trim_s: float,
) -> pd.DataFrame:
    sound = parselmouth.Sound(str(audio_path))
    tier = get_interval_tier(read_textgrid(textgrid_path), tier_name)
    times_s, interval_ids = collect_qc_times(
        tier,
        float(sound.duration),
        qc_step_s,
        edge_trim_s,
        skip_empty,
    )

    rows: list[dict[str, Any]] = []
    tracks: dict[tuple[int, float], np.ndarray] = {}
    for ceiling_hz in FORMANT_CEILINGS_HZ:
        for number_of_formants in FORMANT_COUNTS:
            values, bandwidths = candidate_tracks(
                sound,
                times_s,
                float(ceiling_hz),
                number_of_formants,
                qc_step_s,
            )
            tracks[(ceiling_hz, number_of_formants)] = values
            rows.append(
                {
                    "file": audio_path.name,
                    "maximum_formant_hz": ceiling_hz,
                    "number_of_formants": number_of_formants,
                    **calculate_metrics(values, bandwidths, interval_ids),
                }
            )

    for row in rows:
        key = (int(row["maximum_formant_hz"]), float(row["number_of_formants"]))
        distances = [
            trajectory_distance(tracks[key], tracks[neighbor])
            for neighbor in neighboring_candidates(*key)
        ]
        sensitivity = float(np.mean(distances)) if distances else 1.0
        row["parameter_sensitivity"] = sensitivity
        row["score"] = float(
            row["base_score"] + 1.5 * min(sensitivity / 0.10, 1.0)
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["score", "mean_coverage"], ascending=[True, False]
    )
    ranking.insert(1, "rank", np.arange(1, len(ranking) + 1))
    ranking["recommended"] = ranking["rank"].eq(1)
    ranking["recording_grade"] = recording_grade(ranking.iloc[0])
    return ranking.reset_index(drop=True)


def extract_midpoints(
    audio_path: Path,
    textgrid_path: Path,
    tier_name: str,
    pitch_floor_hz: float,
    pitch_ceiling_hz: float,
    maximum_formant_hz: float,
    number_of_formants: float,
    skip_empty: bool,
) -> list[dict[str, Any]]:
    """Extract exactly one measurement row at 50% of every interval."""
    sound = parselmouth.Sound(str(audio_path))
    tier = get_interval_tier(read_textgrid(textgrid_path), tier_name)
    pitch = sound.to_pitch_ac(
        time_step=0.005,
        pitch_floor=pitch_floor_hz,
        pitch_ceiling=pitch_ceiling_hz,
    )
    intensity = sound.to_intensity(
        minimum_pitch=pitch_floor_hz,
        time_step=0.005,
    )
    formant = sound.to_formant_burg(
        time_step=0.005,
        max_number_of_formants=number_of_formants,
        maximum_formant=maximum_formant_hz,
    )

    rows: list[dict[str, Any]] = []
    for interval_index, interval in enumerate(tier.intervals, start=1):
        label = interval.text.strip()
        if skip_empty and not label:
            continue
        start_s = float(interval.start_time)
        end_s = float(interval.end_time)
        duration_s = end_s - start_s
        if duration_s <= 0:
            continue
        midpoint_s = start_s + duration_s * 0.5
        f0_hz = optional_float(pitch.get_value_at_time(midpoint_s), positive=True)
        intensity_db = optional_float(intensity.get_value(midpoint_s))

        rows.append(
            {
                "file": audio_path.name,
                "textgrid": textgrid_path.name,
                "tier": tier.name,
                "interval_index": interval_index,
                "label": label,
                "start_s": rounded(start_s),
                "end_s": rounded(end_s),
                "midpoint_s": rounded(midpoint_s),
                "midpoint_ms": rounded(midpoint_s * 1000.0, 2),
                "duration_ms": rounded(duration_s * 1000.0, 2),
                "voiced": int(f0_hz is not None),
                "f0_Hz_50": rounded(f0_hz, 2),
                "intensity_dB_50": rounded(intensity_db, 2),
                "F1_Hz_50": rounded(
                    optional_float(
                        formant.get_value_at_time(1, midpoint_s), positive=True
                    ),
                    2,
                ),
                "F2_Hz_50": rounded(
                    optional_float(
                        formant.get_value_at_time(2, midpoint_s), positive=True
                    ),
                    2,
                ),
                "F3_Hz_50": rounded(
                    optional_float(
                        formant.get_value_at_time(3, midpoint_s), positive=True
                    ),
                    2,
                ),
                "selected_maximum_formant_hz": maximum_formant_hz,
                "selected_number_of_formants": number_of_formants,
            }
        )
    return rows


def make_excel(
    measurements: pd.DataFrame,
    audit: pd.DataFrame,
    recommendations: pd.DataFrame,
    metadata: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        measurements.to_excel(writer, sheet_name="midpoint_measurements", index=False)
        if not audit.empty:
            audit.to_excel(writer, sheet_name="formant_parameter_audit", index=False)
            recommendations.to_excel(
                writer, sheet_name="recommended_settings", index=False
            )
        metadata.to_excel(writer, sheet_name="metadata", index=False)
    return buffer.getvalue()


st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
st.title(APP_TITLE)
st.caption("One 50% measurement per interval — Seongwoo Kang")
st.info(
    "The 28-setting audit uses dense internal frames, but the downloaded acoustic "
    "table contains exactly one row per interval at its 50% midpoint."
)

audio_files = st.file_uploader(
    "Audio files (.wav / .aiff / .flac)",
    type=["wav", "aiff", "aif", "flac"],
    accept_multiple_files=True,
)
textgrid_files = st.file_uploader(
    "TextGrid files",
    type=["textgrid"],
    accept_multiple_files=True,
)

st.sidebar.header("Extraction parameters")
tier_name = st.sidebar.text_input("Interval tier", value="segment")
pitch_floor = st.sidebar.number_input("Pitch floor (Hz)", value=75.0, step=5.0)
pitch_ceiling = st.sidebar.number_input("Pitch ceiling (Hz)", value=500.0, step=10.0)
auto_formant = st.sidebar.checkbox("Auto-evaluate formant settings", value=True)

if auto_formant:
    st.sidebar.caption("28 candidates: 4000–5500 Hz × 4.0/4.5/5.0/5.5 formants")
    maximum_formant = 5500.0
    number_of_formants = 5.0
    qc_step_ms = st.sidebar.number_input(
        "Audit frame step (ms)", 5.0, 20.0, 10.0, 5.0
    )
    edge_trim_ms = st.sidebar.number_input(
        "Audit edge trim (ms)", 0.0, 50.0, 10.0, 5.0
    )
else:
    maximum_formant = st.sidebar.number_input(
        "Maximum formant (Hz)", value=5500.0, step=100.0
    )
    number_of_formants = st.sidebar.select_slider(
        "Number of formants",
        options=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        value=5.0,
    )
    qc_step_ms = 10.0
    edge_trim_ms = 10.0

keep_empty = st.sidebar.checkbox("Keep empty intervals", value=False)

validation_errors: list[str] = []
if not tier_name.strip():
    validation_errors.append("Tier name cannot be empty.")
if pitch_floor <= 0 or pitch_floor >= pitch_ceiling:
    validation_errors.append("Pitch floor must be positive and below pitch ceiling.")
for message in validation_errors:
    st.error(message)

audio_map, audio_duplicates = build_upload_map(audio_files or [])
textgrid_map, textgrid_duplicates = build_upload_map(textgrid_files or [])
if audio_duplicates or textgrid_duplicates:
    st.error(
        "Duplicate filename stems: "
        + ", ".join(sorted(set(audio_duplicates + textgrid_duplicates)))
    )

matched = sorted(set(audio_map) & set(textgrid_map))
audio_only = sorted(set(audio_map) - set(textgrid_map))
textgrid_only = sorted(set(textgrid_map) - set(audio_map))
if audio_files or textgrid_files:
    st.write(f"**{len(matched)} matched pair(s)**")
    if audio_only:
        st.warning("Audio without TextGrid: " + ", ".join(audio_only))
    if textgrid_only:
        st.warning("TextGrid without audio: " + ", ".join(textgrid_only))

can_run = bool(
    matched
    and not validation_errors
    and not audio_duplicates
    and not textgrid_duplicates
)
button_label = (
    "Run 28-setting audit + midpoint extraction"
    if auto_formant
    else "Run midpoint extraction"
)

if st.button(button_label, type="primary", disabled=not can_run):
    all_measurements: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    logs: list[str] = []
    progress = st.progress(0.0, text="Preparing…")

    with tempfile.TemporaryDirectory() as temp_directory:
        temp = Path(temp_directory)
        for pair_index, stem in enumerate(matched, start=1):
            audio_upload = audio_map[stem]
            textgrid_upload = textgrid_map[stem]
            audio_path = temp / Path(audio_upload.name).name
            textgrid_path = temp / Path(textgrid_upload.name).name
            audio_path.write_bytes(audio_upload.getvalue())
            textgrid_path.write_bytes(textgrid_upload.getvalue())

            try:
                selected_ceiling = float(maximum_formant)
                selected_count = float(number_of_formants)
                if auto_formant:
                    progress.progress(
                        (pair_index - 1) / len(matched),
                        text=f"Auditing 28 settings for {audio_upload.name}…",
                    )
                    ranking = audit_formant_parameters(
                        audio_path,
                        textgrid_path,
                        tier_name,
                        not keep_empty,
                        float(qc_step_ms) / 1000.0,
                        float(edge_trim_ms) / 1000.0,
                    )
                    all_audits.extend(ranking.to_dict(orient="records"))
                    best = ranking.iloc[0]
                    selected_ceiling = float(best["maximum_formant_hz"])
                    selected_count = float(best["number_of_formants"])
                    logs.append(
                        f"[audit] {audio_upload.name}: {selected_ceiling:.0f} Hz / "
                        f"{selected_count:.1f} formants / {best['recording_grade']}"
                    )

                rows = extract_midpoints(
                    audio_path,
                    textgrid_path,
                    tier_name,
                    float(pitch_floor),
                    float(pitch_ceiling),
                    selected_ceiling,
                    selected_count,
                    not keep_empty,
                )
                if not rows:
                    raise ValueError("No intervals remained after filtering.")
                all_measurements.extend(rows)
                logs.append(f"[ok] {audio_upload.name}: {len(rows)} interval(s)")
            except Exception as exc:
                logs.append(f"[error] {audio_upload.name}: {exc}")

            progress.progress(
                pair_index / len(matched),
                text=f"Processed {pair_index}/{len(matched)} pair(s)",
            )

    progress.empty()
    if all_measurements:
        measurements = pd.DataFrame(all_measurements)
        audit = pd.DataFrame(all_audits)
        recommendations = (
            audit.loc[audit["recommended"]].copy()
            if not audit.empty
            else pd.DataFrame()
        )
        metadata = pd.DataFrame(
            {
                "parameter": [
                    "extracted_at_utc",
                    "measurement_location",
                    "auto_formant_audit",
                    "pitch_floor_hz",
                    "pitch_ceiling_hz",
                    "audit_frame_step_ms",
                    "audit_edge_trim_ms",
                    "warning",
                ],
                "value": [
                    datetime.now(timezone.utc).isoformat(),
                    "50% midpoint; exactly one row per interval",
                    auto_formant,
                    pitch_floor,
                    pitch_ceiling,
                    qc_step_ms,
                    edge_trim_ms,
                    "Audit ranking measures LPC stability, not acoustic ground truth.",
                ],
            }
        )
        st.session_state["midpoint_result"] = {
            "measurements": measurements,
            "audit": audit,
            "recommendations": recommendations,
            "metadata": metadata,
            "logs": logs,
            "csv": measurements.to_csv(index=False, na_rep="").encode("utf-8-sig"),
            "excel": make_excel(measurements, audit, recommendations, metadata),
        }
    else:
        st.session_state.pop("midpoint_result", None)
        st.error("No measurements were extracted.")
        st.code("\n".join(logs), language=None)


result = st.session_state.get("midpoint_result")
if result:
    st.success(
        f"Complete: {len(result['measurements']):,} intervals, "
        "one midpoint row per interval."
    )
    with st.expander("Extraction log"):
        st.code("\n".join(result["logs"]), language=None)

    tab_names = ["Midpoint measurements"]
    if not result["audit"].empty:
        tab_names.append("Formant audit")
    tab_names.append("Metadata")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.dataframe(result["measurements"], use_container_width=True, hide_index=True)

    tab_index = 1
    if not result["audit"].empty:
        with tabs[tab_index]:
            st.warning(
                "F2–F3 proximity is diagnostic only and does not increase the score. "
                "Inspect the top candidates before publication."
            )
            st.subheader("Recommended setting by recording")
            recommendation_columns = [
                "file",
                "maximum_formant_hz",
                "number_of_formants",
                "recording_grade",
                "score",
                "mean_coverage",
                "mean_jump_rate",
                "parameter_sensitivity",
                "F2_F3_close_rate",
            ]
            st.dataframe(
                result["recommendations"][recommendation_columns],
                use_container_width=True,
                hide_index=True,
            )
            selected_file = st.selectbox(
                "Inspect recording",
                result["audit"]["file"].drop_duplicates().tolist(),
            )
            file_audit = result["audit"].loc[
                result["audit"]["file"].eq(selected_file)
            ]
            score_surface = file_audit.pivot(
                index="maximum_formant_hz",
                columns="number_of_formants",
                values="score",
            ).sort_index(ascending=False)
            st.subheader("Score surface — lower is better")
            st.dataframe(score_surface, use_container_width=True)
            st.subheader("All 28 candidates")
            st.dataframe(
                file_audit.sort_values("rank"),
                use_container_width=True,
                hide_index=True,
            )
        tab_index += 1

    with tabs[tab_index]:
        st.dataframe(result["metadata"], use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download midpoint CSV",
            result["csv"],
            "midpoint_acoustics.csv",
            "text/csv",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Download Excel with audit",
            result["excel"],
            "midpoint_acoustics_with_formant_audit.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
elif not audio_files or not textgrid_files:
    st.info("Upload at least one matching audio/TextGrid pair to begin.")
