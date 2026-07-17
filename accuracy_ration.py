"""
accuracy_analysis.py

Compares three sets of anthropometric measurements:
  - Actual   : ground-truth measurements taken by hand with a ruler (cm)
  - Polycam  : measurements taken manually on a Polycam 3D reconstruction (mm)
  - Python   : measurements produced automatically by the MediaPipe pipeline (mm)

For every feature that is present in ALL three sources, and every person that
has data in ALL three sources, it computes:
  - MAE   (Mean Absolute Error)
  - RMSE  (Root Mean Square Error)
  - MAPE  (Mean Absolute Percentage Error)
  - Mean Bias (signed average error -> over/under-estimation)
  - SD of Error (consistency of the error, independent of its average size)
  - Pearson r (how well each method tracks the true value)
  - Paired t-test (Polycam vs Actual, Python vs Actual, Polycam vs Python)

Results are written to an Excel workbook with one sheet per level of
aggregation (overall / by feature / by person), plus the merged raw data.

Requirements:
    pip install pandas openpyxl scipy --break-system-packages

Usage:
    python accuracy_analysis.py
(edit the three file paths and the name/feature maps below to match your data)
"""

import pandas as pd
import numpy as np
from scipy import stats

# =====================================================================
# 1. CONFIG - EDIT THESE TO MATCH YOUR FILES
# =====================================================================

ACTUAL_XLSX = "actual_values.xlsx"          # ground truth, values in CM
COMBINED_XLSX = "combined.xlsx"             # Polycam values, already in MM
PYTHON_CSV = "measurements10_python.csv"    # MediaPipe values, already in MM

OUTPUT_XLSX = "accuracy_analysis_results_ratio.xlsx"

# Canonical person names -> how each source spells them.
# Add/remove people here; a person is only included if they appear in ALL
# THREE of the dicts below.
ACTUAL_NAME_MAP = {
    "Yogesh": "Yogesh", "sonu": "Sonu", "Sunayana": "Sunayana",
    "VAN": "VAN", "Khyati": "Khyati", "Pratibha": "Pratibha",
}
COMBINED_NAME_MAP = {
    "yogesh": "Yogesh", "sunayana": "Sunayana", "SONU": "Sonu",
    "VAN": "VAN", "Khyati": "Khyati", "pratibha": "Pratibha",
}
PYTHON_NAME_MAP = {
    "yogesh": "Yogesh", "sunauyna": "Sunayana", "sonu1": "Sonu",
    "van": "VAN", "khyati": "Khyati", "pratibha": "Pratibha",
}

# Canonical feature names -> how each source spells them.
# Only features present (with a non-null value) in ALL THREE sources for a
# given person/feature pair are used.
ACTUAL_FEATURE_MAP = {
    "left_eye_width": "left_eye_width", "right_eye_width": "right_eye_width",
    "left_eye_height": "left_eye_height", "right_eye_height": "right_eye_height",
    "inner_canthal_distance": "inner_canthal_distance",
    "outer_canthal_distance": "outer_canthal_distance",
    "interpupillary_distance": "interpupillary_distance",
    "nose_width": "nose_width", "nose_height": "nose_height",
    "mouth_width": "mouth_width", "mouth_height": "mouth_height",
    "face_width": "face_width", "face_height": "face_height",
    "jaw_width": "jaw_width",
    "hand_length": "hand_length", "hand_width": "hand_width",
    "palm_w": "palm_width", "palm_h": "palm_height",
    "index": "index", "middle": "middle", "ring": "ring",
    "pinky": "pinky", "thumb": "thumb",
}
COMBINED_FEATURE_MAP = {
    "left_eye_width": "left_eye_width", "right_eye_width": "right_eye_width",
    "left_eye_height": "left_eye_height", "right_eye_height": "right_eye_height",
    "inner_canthal_distance": "inner_canthal_distance",
    "outer_canthal_distance": "outer_canthal_distance",
    "interpupillary_distance": "interpupillary_distance",
    "nose_width": "nose_width", "nose_height": "nose_height",
    "mouth_width": "mouth_width", "mouth_height": "mouth_height",
    "face_width": "face_width", "face_height": "face_height",
    "jaw_width": "jaw_width",
    "hand length": "hand_length", "hand width": "hand_width",
    "palm w": "palm_width", "palm h": "palm_height",
    "index": "index", "middle": "middle", "ring": "ring",
    "pinky": "pinky", "thumb": "thumb",
}
PYTHON_FEATURE_MAP = {
    "left_eye_width": "left_eye_width", "right_eye_width": "right_eye_width",
    "left_eye_height": "left_eye_height", "right_eye_height": "right_eye_height",
    "inner_canthal_distance": "inner_canthal_distance",
    "outer_canthal_distance": "outer_canthal_distance",
    "interpupillary_distance": "interpupillary_distance",
    "nose_width": "nose_width", "nose_height": "nose_height",
    "mouth_width": "mouth_width", "mouth_height": "mouth_height",
    "face_width": "face_width", "face_height": "face_height",
    "jaw_width": "jaw_width",
    "hand_length_mm": "hand_length", "hand_width_mm": "hand_width",
    "palm_width_mm": "palm_width", "palm_height_mm": "palm_height",
    "thumb_mm": "thumb", "index_mm": "index", "middle_mm": "middle",
    "ring_mm": "ring", "pinky_mm": "pinky",
}

ACTUAL_UNIT_TO_MM = 10.0   # actual_values.xlsx is in cm -> multiply by 10
COMBINED_UNIT_TO_MM = 1.0  # combined.xlsx is already in mm
PYTHON_UNIT_TO_MM = 1.0    # python csv is already in mm


# =====================================================================
# 2. LOAD + MERGE
# =====================================================================

def load_long(path, is_csv, name_map, feature_map, unit_to_mm,
              row_is_feature, name_col=None, feature_col=None):
    """
    Returns a long dataframe: feature, candidate, value(mm)

    row_is_feature=True  -> file has one row per FEATURE, one column per PERSON
                             (used for actual_values.xlsx / combined.xlsx)
    row_is_feature=False -> file has one row per PERSON, one column per FEATURE
                             (used for the python csv)
    """
    df = pd.read_csv(path) if is_csv else pd.read_excel(path)
    long_rows = []

    if row_is_feature:
        for _, row in df.iterrows():
            feat_raw = row[feature_col]
            if feat_raw not in feature_map:
                continue
            feat = feature_map[feat_raw]
            for raw_name, canon_name in name_map.items():
                if raw_name not in df.columns:
                    continue
                val = row[raw_name]
                if pd.notna(val):
                    long_rows.append((feat, canon_name, float(val) * unit_to_mm))
    else:
        for _, row in df.iterrows():
            name_raw = row[name_col]
            if name_raw not in name_map:
                continue
            canon_name = name_map[name_raw]
            for raw_feat, feat in feature_map.items():
                if raw_feat not in df.columns:
                    continue
                val = row[raw_feat]
                if pd.notna(val):
                    long_rows.append((feat, canon_name, float(val) * unit_to_mm))

    return pd.DataFrame(long_rows, columns=["feature", "candidate", "value_mm"])


def build_merged_table():
    actual_long = load_long(ACTUAL_XLSX, is_csv=False, name_map=ACTUAL_NAME_MAP,
                             feature_map=ACTUAL_FEATURE_MAP, unit_to_mm=ACTUAL_UNIT_TO_MM,
                             row_is_feature=True, feature_col="Measurement")
    combined_long = load_long(COMBINED_XLSX, is_csv=False, name_map=COMBINED_NAME_MAP,
                               feature_map=COMBINED_FEATURE_MAP, unit_to_mm=COMBINED_UNIT_TO_MM,
                               row_is_feature=True, feature_col="name")
    python_long = load_long(PYTHON_CSV, is_csv=True, name_map=PYTHON_NAME_MAP,
                             feature_map=PYTHON_FEATURE_MAP, unit_to_mm=PYTHON_UNIT_TO_MM,
                             row_is_feature=False, name_col="name")

    actual_p = actual_long.pivot(index=["feature", "candidate"], columns=[], values="value_mm") \
        if False else actual_long.rename(columns={"value_mm": "Actual_mm"})
    combined_p = combined_long.rename(columns={"value_mm": "Polycam_mm"})
    python_p = python_long.rename(columns={"value_mm": "Python_mm"})

    merged = actual_p.merge(combined_p, on=["feature", "candidate"], how="inner") \
                      .merge(python_p, on=["feature", "candidate"], how="inner")
    # inner joins on all three -> only feature/candidate pairs present in ALL THREE remain
    return merged


# =====================================================================
# 3. METRICS
# =====================================================================

def compute_metrics(df, actual_col, method_col):
    """df must have columns [actual_col, method_col] with matched rows (paired data)."""
    a = df[actual_col].to_numpy(dtype=float)
    m = df[method_col].to_numpy(dtype=float)
    err = m - a
    abs_err = np.abs(err)
    pct_err = np.where(a != 0, abs_err / a * 100.0, np.nan)

    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.nanmean(pct_err)
    bias = np.mean(err)
    sd_err = np.std(err, ddof=1) if len(err) > 1 else np.nan
    r, _ = stats.pearsonr(m, a) if len(a) > 1 else (np.nan, np.nan)

    return {
        "n": len(df), "MAE": mae, "RMSE": rmse, "MAPE_%": mape,
        "Mean_Bias": bias, "SD_of_Error": sd_err, "Pearson_r": r,
    }


def compute_ratio_stats(df, actual_col, method_col):
    """
    Ratio = method_value / actual_value for each row.
    A ratio of 1.0 means perfect agreement; 0.9 means the method reads
    ~10% low; 1.1 means it reads ~10% high. This is the number you'd
    multiply the method's raw output by (1/mean_ratio) to correct it.
    """
    a = df[actual_col].to_numpy(dtype=float)
    m = df[method_col].to_numpy(dtype=float)
    ratio = np.where(a != 0, m / a, np.nan)
    ratio = ratio[~np.isnan(ratio)]

    return {
        "n": len(ratio),
        "Mean_Ratio": np.mean(ratio) if len(ratio) else np.nan,
        "Median_Ratio": np.median(ratio) if len(ratio) else np.nan,
        "SD_Ratio": np.std(ratio, ddof=1) if len(ratio) > 1 else np.nan,
        "Suggested_Correction_Factor": (1.0 / np.mean(ratio)) if len(ratio) and np.mean(ratio) != 0 else np.nan,
    }



def paired_tests(df, actual_col):
    """Paired t-tests: each method vs Actual (bias test), and Polycam vs Python (method comparison)."""
    a = df[actual_col].to_numpy(dtype=float)
    poly = df["Polycam_mm"].to_numpy(dtype=float)
    py = df["Python_mm"].to_numpy(dtype=float)

    t_poly_vs_actual = stats.ttest_rel(poly, a)
    t_py_vs_actual = stats.ttest_rel(py, a)
    t_poly_vs_py_abserr = stats.ttest_rel(np.abs(poly - a), np.abs(py - a))

    return pd.DataFrame([
        {"comparison": "Polycam vs Actual (bias != 0?)", "n": len(df),
         "t_stat": t_poly_vs_actual.statistic, "p_value": t_poly_vs_actual.pvalue},
        {"comparison": "Python vs Actual (bias != 0?)", "n": len(df),
         "t_stat": t_py_vs_actual.statistic, "p_value": t_py_vs_actual.pvalue},
        {"comparison": "Polycam |error| vs Python |error|", "n": len(df),
         "t_stat": t_poly_vs_py_abserr.statistic, "p_value": t_poly_vs_py_abserr.pvalue},
    ])


# =====================================================================
# 4. MAIN
# =====================================================================

def main():
    merged = build_merged_table()
    if merged.empty:
        raise SystemExit(
            "No overlapping feature/candidate pairs found across all three files. "
            "Check the *_NAME_MAP and *_FEATURE_MAP dictionaries at the top of this script."
        )

    print(f"Merged {len(merged)} feature/candidate pairs "
          f"({merged['candidate'].nunique()} people, {merged['feature'].nunique()} features).")

    # ---- overall ----
    overall_rows = []
    for method_col, label in [("Polycam_mm", "Polycam"), ("Python_mm", "Python")]:
        m = compute_metrics(merged, "Actual_mm", method_col)
        m["method"] = label
        overall_rows.append(m)
    overall_df = pd.DataFrame(overall_rows).set_index("method")
    overall_df = overall_df[["n", "MAE", "RMSE", "MAPE_%", "Mean_Bias", "SD_of_Error", "Pearson_r"]]

    # ---- by feature ----
    feature_rows = []
    for feat, sub in merged.groupby("feature"):
        row = {"feature": feat}
        for method_col, prefix in [("Polycam_mm", "Polycam"), ("Python_mm", "Python")]:
            m = compute_metrics(sub, "Actual_mm", method_col)
            for k, v in m.items():
                if k != "n":
                    row[f"{prefix}_{k}"] = v
        row["n"] = len(sub)
        row["more_accurate"] = "Polycam" if row["Polycam_MAE"] < row["Python_MAE"] else "Python"
        feature_rows.append(row)
    by_feature_df = pd.DataFrame(feature_rows).sort_values("feature").reset_index(drop=True)

    # ---- by person ----
    person_rows = []
    for cand, sub in merged.groupby("candidate"):
        row = {"candidate": cand}
        for method_col, prefix in [("Polycam_mm", "Polycam"), ("Python_mm", "Python")]:
            m = compute_metrics(sub, "Actual_mm", method_col)
            for k, v in m.items():
                if k != "n":
                    row[f"{prefix}_{k}"] = v
        row["n"] = len(sub)
        row["more_accurate"] = "Polycam" if row["Polycam_MAE"] < row["Python_MAE"] else "Python"
        person_rows.append(row)
    by_person_df = pd.DataFrame(person_rows).sort_values("candidate").reset_index(drop=True)

    # ---- ratios by feature (measured / actual, per feature) ----
    ratio_feature_rows = []
    for feat, sub in merged.groupby("feature"):
        row = {"feature": feat}
        for method_col, prefix in [("Polycam_mm", "Polycam"), ("Python_mm", "Python")]:
            rstats = compute_ratio_stats(sub, "Actual_mm", method_col)
            for k, v in rstats.items():
                if k != "n":
                    row[f"{prefix}_{k}"] = v
        row["n"] = len(sub)
        ratio_feature_rows.append(row)
    ratio_by_feature_df = pd.DataFrame(ratio_feature_rows).sort_values("feature").reset_index(drop=True)

    # ---- ratios by candidate (measured / actual, per person, across all features) ----
    ratio_person_rows = []
    for cand, sub in merged.groupby("candidate"):
        row = {"candidate": cand}
        for method_col, prefix in [("Polycam_mm", "Polycam"), ("Python_mm", "Python")]:
            rstats = compute_ratio_stats(sub, "Actual_mm", method_col)
            for k, v in rstats.items():
                if k != "n":
                    row[f"{prefix}_{k}"] = v
        row["n"] = len(sub)
        ratio_person_rows.append(row)
    ratio_by_person_df = pd.DataFrame(ratio_person_rows).sort_values("candidate").reset_index(drop=True)

    # ---- paired statistical tests (overall) ----
    tests_df = paired_tests(merged, "Actual_mm")

    # ---- write results ----
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="Merged Data (mm)", index=False)
        overall_df.round(4).to_excel(writer, sheet_name="Overall Summary")
        by_feature_df.round(4).to_excel(writer, sheet_name="Summary by Feature", index=False)
        by_person_df.round(4).to_excel(writer, sheet_name="Summary by Candidate", index=False)
        ratio_by_feature_df.round(4).to_excel(writer, sheet_name="Ratios by Feature", index=False)
        ratio_by_person_df.round(4).to_excel(writer, sheet_name="Ratios by Candidate", index=False)
        tests_df.to_excel(writer, sheet_name="Paired t-tests", index=False)

    print(f"\nSaved results to: {OUTPUT_XLSX}")
    print("\n=== Overall Summary ===")
    print(overall_df.round(3).to_string())
    print("\n=== Ratios by Feature (measured / actual) ===")
    print(ratio_by_feature_df.round(3).to_string(index=False))
    print("\n=== Paired t-tests ===")
    print(tests_df.to_string(index=False))


if __name__ == "__main__":
    main()