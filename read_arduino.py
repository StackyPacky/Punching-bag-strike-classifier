import serial
import csv
import os
import math
import time
from collections import OrderedDict

import pandas as pd
import joblib


PORT = "COM3"
BAUD = 115200

raw_file_path = r"C:\Users\vladb\OneDrive\Documents\punch_data_3piezo.csv"
features_file_path = r"C:\Users\vladb\OneDrive\Documents\punch_features_3piezo.csv"
model_path = r"C:\Users\vladb\OneDrive\Documents\punch_model.pkl"

start_threshold = 90
continue_threshold = 60
gap_ms = 320

# ---------------------------------
# DATA COLLECTION SETTINGS
# ---------------------------------
# While collecting right kick data, leave this as "right_kick"
# When you want normal unlabeled collection again, change it to "unlabeled"
collection_label = "unlabeled"

# Labels that should NOT get a power score
no_power_labels = {"right_kick"}


# ---------------------------
# Parsing / math helpers
# ---------------------------

def safe_div(a, b):
    return a / b if b != 0 else 0.0


def argmax3(a, b, c):
    vals = [a, b, c]
    return vals.index(max(vals)) + 1  # 1, 2, or 3


def second_largest_of_three(a, b, c):
    vals = sorted([a, b, c], reverse=True)
    return vals[1]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_to_unit(value, low, high):
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def parse_row(parts):
    return {
        "time_ms": int(parts[0]),
        "p1": int(parts[1]),
        "p2": int(parts[2]),
        "p3": int(parts[3]),
        "ax": int(parts[4]),
        "ay": int(parts[5]),
        "az": int(parts[6]),
    }


def accel_mag(row):
    return math.sqrt(row["ax"] ** 2 + row["ay"] ** 2 + row["az"] ** 2)


def summarize_series(values, times, name_prefix):
    """
    Build shape features for one signal series.
    """
    n = len(values)
    max_val = max(values)
    min_val = min(values)
    sum_val = sum(values)
    mean_val = sum_val / n

    peak_idx = values.index(max_val)
    peak_time = times[peak_idx]
    start_time = times[0]
    end_time = times[-1]

    time_to_peak_ms = peak_time - start_time
    time_after_peak_ms = end_time - peak_time
    peak_pos_frac = peak_idx / max(n - 1, 1)

    first_val = values[0]
    last_val = values[-1]

    first_half_end = max(1, n // 2)
    first_half_sum = sum(values[:first_half_end])
    second_half_sum = sum(values[first_half_end:])

    pre_peak_sum = sum(values[:peak_idx + 1])
    post_peak_sum = sum(values[peak_idx:])

    rise_slope = safe_div(max_val - first_val, max(time_to_peak_ms, 1))
    fall_slope = safe_div(max_val - last_val, max(time_after_peak_ms, 1))

    above_50 = sum(1 for v in values if v >= 0.50 * max_val)
    above_75 = sum(1 for v in values if v >= 0.75 * max_val)
    above_90 = sum(1 for v in values if v >= 0.90 * max_val)

    local_start = max(0, peak_idx - 1)
    local_end = min(n, peak_idx + 2)
    local_peak_window_sum = sum(values[local_start:local_end])
    peak_concentration = safe_div(local_peak_window_sum, max(sum_val, 1))

    left_neighbor = values[peak_idx - 1] if peak_idx - 1 >= 0 else values[peak_idx]
    right_neighbor = values[peak_idx + 1] if peak_idx + 1 < n else values[peak_idx]
    sharper_than_neighbors = max_val - ((left_neighbor + right_neighbor) / 2)

    return OrderedDict({
        f"{name_prefix}_max": max_val,
        f"{name_prefix}_min": min_val,
        f"{name_prefix}_sum": sum_val,
        f"{name_prefix}_mean": mean_val,

        f"{name_prefix}_first": first_val,
        f"{name_prefix}_last": last_val,

        f"{name_prefix}_peak_idx": peak_idx,
        f"{name_prefix}_peak_time_ms": time_to_peak_ms,
        f"{name_prefix}_peak_pos_frac": peak_pos_frac,

        f"{name_prefix}_time_after_peak_ms": time_after_peak_ms,

        f"{name_prefix}_first_half_sum": first_half_sum,
        f"{name_prefix}_second_half_sum": second_half_sum,

        f"{name_prefix}_pre_peak_sum": pre_peak_sum,
        f"{name_prefix}_post_peak_sum": post_peak_sum,

        f"{name_prefix}_rise_slope": rise_slope,
        f"{name_prefix}_fall_slope": fall_slope,

        f"{name_prefix}_rows_above_50pct": above_50,
        f"{name_prefix}_rows_above_75pct": above_75,
        f"{name_prefix}_rows_above_90pct": above_90,

        f"{name_prefix}_peak_concentration": peak_concentration,
        f"{name_prefix}_peak_sharpness": sharper_than_neighbors,
    })


# ---------------------------
# Feature extraction
# ---------------------------

def condense_event(current_event_rows):
    parsed_rows = [parse_row(r) for r in current_event_rows]

    num_rows = len(parsed_rows)
    times = [r["time_ms"] for r in parsed_rows]
    start_time = times[0]
    end_time = times[-1]
    duration_ms = end_time - start_time

    p1 = [r["p1"] for r in parsed_rows]
    p2 = [r["p2"] for r in parsed_rows]
    p3 = [r["p3"] for r in parsed_rows]

    ax = [r["ax"] for r in parsed_rows]
    ay = [r["ay"] for r in parsed_rows]
    az = [r["az"] for r in parsed_rows]
    amag = [accel_mag(r) for r in parsed_rows]

    max_p1 = max(p1)
    max_p2 = max(p2)
    max_p3 = max(p3)

    sum_p1 = sum(p1)
    sum_p2 = sum(p2)
    sum_p3 = sum(p3)

    total_piezo_sum = sum_p1 + sum_p2 + sum_p3
    total_piezo_max = max_p1 + max_p2 + max_p3

    dominant_peak_sensor = argmax3(max_p1, max_p2, max_p3)
    dominant_sum_sensor = argmax3(sum_p1, sum_p2, sum_p3)

    dominant_peak_value = max(max_p1, max_p2, max_p3)
    dominant_sum_value = max(sum_p1, sum_p2, sum_p3)

    second_peak_value = second_largest_of_three(max_p1, max_p2, max_p3)
    second_sum_value = second_largest_of_three(sum_p1, sum_p2, sum_p3)

    overall_peak_value = 0
    overall_peak_idx = 0
    overall_peak_sensor = 1

    for i in range(num_rows):
        row_peak = max(p1[i], p2[i], p3[i])
        if row_peak > overall_peak_value:
            overall_peak_value = row_peak
            overall_peak_idx = i
            overall_peak_sensor = argmax3(p1[i], p2[i], p3[i])

    overall_peak_time_ms = times[overall_peak_idx] - start_time

    features = OrderedDict()

    features["num_rows"] = num_rows
    features["duration_ms"] = duration_ms
    features["rows_per_10ms"] = safe_div(num_rows, max(duration_ms, 1)) * 10.0

    features["total_piezo_sum"] = total_piezo_sum
    features["total_piezo_max_sum"] = total_piezo_max

    features["p1_sum_share"] = safe_div(sum_p1, max(total_piezo_sum, 1))
    features["p2_sum_share"] = safe_div(sum_p2, max(total_piezo_sum, 1))
    features["p3_sum_share"] = safe_div(sum_p3, max(total_piezo_sum, 1))

    features["p1_peak_share"] = safe_div(max_p1, max(total_piezo_max, 1))
    features["p2_peak_share"] = safe_div(max_p2, max(total_piezo_max, 1))
    features["p3_peak_share"] = safe_div(max_p3, max(total_piezo_max, 1))

    features["dominant_peak_sensor"] = dominant_peak_sensor
    features["dominant_sum_sensor"] = dominant_sum_sensor

    features["dominant_peak_margin"] = dominant_peak_value - second_peak_value
    features["dominant_sum_margin"] = dominant_sum_value - second_sum_value

    features["max_p1_minus_p2"] = max_p1 - max_p2
    features["max_p1_minus_p3"] = max_p1 - max_p3
    features["max_p2_minus_p3"] = max_p2 - max_p3

    features["sum_p1_minus_p2"] = sum_p1 - sum_p2
    features["sum_p1_minus_p3"] = sum_p1 - sum_p3
    features["sum_p2_minus_p3"] = sum_p2 - sum_p3

    features["p1_over_p2_peak"] = safe_div(max_p1, max(max_p2, 1))
    features["p1_over_p3_peak"] = safe_div(max_p1, max(max_p3, 1))
    features["p2_over_p3_peak"] = safe_div(max_p2, max(max_p3, 1))

    features["p1_over_p2_sum"] = safe_div(sum_p1, max(sum_p2, 1))
    features["p1_over_p3_sum"] = safe_div(sum_p1, max(sum_p3, 1))
    features["p2_over_p3_sum"] = safe_div(sum_p2, max(sum_p3, 1))

    features["overall_peak_value"] = overall_peak_value
    features["overall_peak_idx"] = overall_peak_idx
    features["overall_peak_time_ms"] = overall_peak_time_ms
    features["overall_peak_pos_frac"] = safe_div(overall_peak_idx, max(num_rows - 1, 1))
    features["overall_peak_sensor"] = overall_peak_sensor

    features["p1_at_overall_peak"] = p1[overall_peak_idx]
    features["p2_at_overall_peak"] = p2[overall_peak_idx]
    features["p3_at_overall_peak"] = p3[overall_peak_idx]

    features["ax_at_overall_peak"] = ax[overall_peak_idx]
    features["ay_at_overall_peak"] = ay[overall_peak_idx]
    features["az_at_overall_peak"] = az[overall_peak_idx]
    features["amag_at_overall_peak"] = amag[overall_peak_idx]

    features.update(summarize_series(p1, times, "p1"))
    features.update(summarize_series(p2, times, "p2"))
    features.update(summarize_series(p3, times, "p3"))

    p1_peak_idx = features["p1_peak_idx"]
    p2_peak_idx = features["p2_peak_idx"]
    p3_peak_idx = features["p3_peak_idx"]

    features["peak_idx_diff_p1_p2"] = p1_peak_idx - p2_peak_idx
    features["peak_idx_diff_p1_p3"] = p1_peak_idx - p3_peak_idx
    features["peak_idx_diff_p2_p3"] = p2_peak_idx - p3_peak_idx

    features["ax_abs_max"] = max(abs(v) for v in ax)
    features["ay_abs_max"] = max(abs(v) for v in ay)
    features["az_abs_max"] = max(abs(v) for v in az)

    features["ax_mean"] = sum(ax) / num_rows
    features["ay_mean"] = sum(ay) / num_rows
    features["az_mean"] = sum(az) / num_rows

    features["amag_max"] = max(amag)
    features["amag_mean"] = sum(amag) / num_rows
    features["amag_sum"] = sum(amag)

    amag_peak_idx = amag.index(max(amag))
    features["amag_peak_idx"] = amag_peak_idx
    features["amag_peak_time_ms"] = times[amag_peak_idx] - start_time
    features["amag_peak_pos_frac"] = safe_div(amag_peak_idx, max(num_rows - 1, 1))
    features["amag_peak_minus_overall_piezo_peak_idx"] = amag_peak_idx - overall_peak_idx

    return features


# ---------------------------
# Prediction
# ---------------------------

def predict_punch(model, features):
    feature_df = pd.DataFrame([features])

    if hasattr(model, "feature_names_in_"):
        needed_cols = list(model.feature_names_in_)
        for col in needed_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df[needed_cols]

    prediction = model.predict(feature_df)[0]
    return prediction


def compute_power_score(prediction, features):
    """
    Rule-based power score from 1 to 10.
    Uses only piezo-based event features.
    Returns None for labels that should not have power.
    """

    label_key = str(prediction).strip().lower()
    if label_key in no_power_labels:
        return None

    peak = features["overall_peak_value"]
    total = features["total_piezo_sum"]

    power_ranges = {
        "jab": {
            "peak_low": 220,
            "peak_high": 520,
            "sum_low": 300,
            "sum_high": 1000,
        },
        "cross": {
            "peak_low": 400,
            "peak_high": 1000,
            "sum_low": 600,
            "sum_high": 1900,
        },
        "left_hook": {
            "peak_low": 640,
            "peak_high": 1250,
            "sum_low": 820,
            "sum_high": 2100,
        },
        "right_hook": {
            "peak_low": 640,
            "peak_high": 1250,
            "sum_low": 820,
            "sum_high": 2100,
        },
    }

    default_ranges = {
        "peak_low": 130,
        "peak_high": 500,
        "sum_low": 220,
        "sum_high": 1100,
    }

    cfg = power_ranges.get(label_key, default_ranges)

    peak_score = normalize_to_unit(peak, cfg["peak_low"], cfg["peak_high"])
    sum_score = normalize_to_unit(total, cfg["sum_low"], cfg["sum_high"])

    combined = (0.7 * peak_score) + (0.3 * sum_score)

    power_1_to_10 = round(1 + combined * 9)

    return int(clamp(power_1_to_10, 1, 10))


# ---------------------------
# CSV helpers
# ---------------------------

def ensure_raw_header(writer):
    writer.writerow(["time_ms", "p1", "p2", "p3", "ax", "ay", "az", "event_id"])


def ensure_features_header(writer, feature_keys):
    writer.writerow(["event_id"] + list(feature_keys) + ["label"])


def write_feature_row(writer, event_id, features, label="unlabeled"):
    writer.writerow([event_id] + [features[k] for k in features.keys()] + [label])


# ---------------------------
# Reader
# ---------------------------

def finish_event(event_id, current_event_rows, feat_writer, feat_f, punch_callback, model, features_header_written_this_run):
    features = condense_event(current_event_rows)

    if not features_header_written_this_run:
        ensure_features_header(feat_writer, features.keys())
        features_header_written_this_run = True

    prediction = predict_punch(model, features)
    power_score = compute_power_score(prediction, features)

    print(f"\nEvent {event_id} ended")
    print(f"Rows collected: {len(current_event_rows)}")
    print("Condensed features:")
    print(features)
    print(f"ML Prediction: {prediction}")

    if power_score is None:
        print("Power: N/A")
    else:
        print(f"Power: {power_score}/10")

    print(f"Saved label: {collection_label}")

    write_feature_row(feat_writer, event_id, features, label=collection_label)
    feat_f.flush()

    if punch_callback:
        punch_callback(prediction, power_score)

    return features_header_written_this_run


def run_reader(punch_callback=None):
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        print("Run train_model.py first.")
        return

    model = joblib.load(model_path)
    print("Model loaded.")
    print(f"Collection label is set to: {collection_label}")

    ser = serial.Serial(PORT, BAUD, timeout=0.05)

    raw_file_exists = os.path.isfile(raw_file_path)
    features_file_exists = os.path.isfile(features_file_path)

    event_id = 0
    current_event_rows = []
    event_active = False
    last_hit_wall_time = None
    last_event_time_ms = None
    features_header_written_this_run = features_file_exists

    with open(raw_file_path, "a", newline="") as raw_f, open(features_file_path, "a", newline="") as feat_f:
        raw_writer = csv.writer(raw_f)
        feat_writer = csv.writer(feat_f)

        if not raw_file_exists:
            ensure_raw_header(raw_writer)

        raw_writer.writerow(["NEW_SESSION"])
        raw_f.flush()

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            now_wall_ms = time.time() * 1000

            if event_active and last_hit_wall_time is not None and current_event_rows:
                if now_wall_ms - last_hit_wall_time > gap_ms:
                    features_header_written_this_run = finish_event(
                        event_id,
                        current_event_rows,
                        feat_writer,
                        feat_f,
                        punch_callback,
                        model,
                        features_header_written_this_run
                    )

                    current_event_rows = []
                    event_active = False
                    last_event_time_ms = None
                    last_hit_wall_time = None

            if not line:
                continue

            if line.startswith("time_ms"):
                continue

            parts = line.split(",")

            if len(parts) != 7:
                continue

            try:
                row = parse_row(parts)
            except ValueError:
                continue

            current_time_ms = row["time_ms"]
            max_piezo = max(row["p1"], row["p2"], row["p3"])

            if event_active and last_event_time_ms is not None:
                if current_time_ms - last_event_time_ms > gap_ms and current_event_rows:
                    features_header_written_this_run = finish_event(
                        event_id,
                        current_event_rows,
                        feat_writer,
                        feat_f,
                        punch_callback,
                        model,
                        features_header_written_this_run
                    )

                    current_event_rows = []
                    event_active = False
                    last_event_time_ms = None
                    last_hit_wall_time = None

            should_add_to_event = False

            if not event_active:
                if max_piezo > start_threshold:
                    event_id += 1
                    event_active = True
                    current_event_rows = []
                    should_add_to_event = True
            else:
                if max_piezo > continue_threshold:
                    should_add_to_event = True

            if should_add_to_event:
                current_event_rows.append(parts)
                last_event_time_ms = current_time_ms
                last_hit_wall_time = now_wall_ms

                print(f"timer reset: max_piezo={max_piezo}, time_ms={row['time_ms']}")
                print(parts + [event_id])

                raw_writer.writerow(parts + [event_id])
                raw_f.flush()
            else:
                raw_writer.writerow(parts + [0])
                raw_f.flush()


if __name__ == "__main__":
    run_reader()