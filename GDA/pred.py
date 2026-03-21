"""
pred.py — Painting classifier using pre-trained GDA (Gaussian Discriminant Analysis).

Only imports: standard libraries, numpy, pandas.
Loads model parameters from gda_model.npz and preprocessing_params.json.
"""

import json
import re
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Locate model files relative to this script
# ═══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = _SCRIPT_DIR / "gda_model.npz"
_PARAMS_PATH = _SCRIPT_DIR / "preprocessing_params.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing utilities (identical to training)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_likert(value) -> float:
    if pd.isna(value):
        return np.nan
    m = re.match(r"^(\d+)", str(value).strip())
    return float(m.group(1)) if m else np.nan


def _parse_price(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).lower().strip().replace(",", "")
    m = re.search(r"(\d+\.?\d*)\s*(?:-|\u2013|to)\s*(\d+\.?\d*)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    for pat, mult in [
        (r"(\d+\.?\d*)\s*(?:billion|bn|b)\b", 1e9),
        (r"(\d+\.?\d*)\s*(?:million|m)\b", 1e6),
        (r"(\d+\.?\d*)\s*k\b", 1e3),
    ]:
        m = re.search(pat, s)
        if m:
            return float(m.group(1)) * mult
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return np.nan


def _binarize_multi_label(series: pd.Series, categories: list) -> pd.DataFrame:
    result = pd.DataFrame(0, index=series.index, columns=categories)
    for idx, val in series.items():
        if pd.notna(val):
            for cat in categories:
                if cat in str(val):
                    result.at[idx, cat] = 1
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TF-IDF (identical to training)
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list:
    if not isinstance(text, str) or pd.isna(text):
        return []
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


def _texts_to_tfidf(texts: list, vocab: list, idf: dict) -> np.ndarray:
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        tokens = _tokenize(text)
        if not tokens:
            continue
        tf = Counter(tokens)
        for w, c in tf.items():
            if w in vi:
                X[i, vi[w]] = (1 + np.log(c)) * idf.get(w, 0)
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Main prediction function
# ═══════════════════════════════════════════════════════════════════════════════

def predict_all(csv_filename: str) -> list[str]:
    """
    Load a CSV file and return painting predictions for every row.

    Parameters
    ----------
    csv_filename : path to a CSV with the same column structure as the training data

    Returns
    -------
    list of painting names, one per row
    """
    # ── load model artifacts ──
    model_data = np.load(_MODEL_PATH)
    with open(_PARAMS_PATH, "r") as f:
        params = json.load(f)

    class_names = params["class_names"]
    column_map = params["column_map"]
    likert_cols = params["likert_cols"]
    text_cols = params["text_cols"]
    numeric_cols = params["numeric_cols"]
    binary_cols = params["binary_cols"]
    numeric_medians = params["numeric_medians"]
    vocab = params["vocab"]
    idf = params["idf"]
    use_qda = params["use_qda"]
    season_cats = params["season_categories"]
    room_cats = params["room_categories"]
    comp_cats = params["companion_categories"]

    class_priors = model_data["class_priors"]
    class_means = model_data["class_means"]
    shared_cov_inv = model_data["shared_cov_inv"]
    shared_cov_logdet = float(model_data["shared_cov_logdet"][0])
    feature_means = model_data["feature_means"]
    feature_stds = model_data["feature_stds"]

    K = len(class_names)

    if use_qda:
        pc_cov_invs = [model_data[f"cov_inv_{k}"] for k in range(K)]
        pc_cov_logdets = [float(model_data[f"cov_logdet_{k}"][0]) for k in range(K)]

    # ── load & preprocess data ──
    df = pd.read_csv(csv_filename).rename(columns=column_map)

    for col in likert_cols:
        if col in df.columns:
            df[col] = df[col].apply(_extract_likert)

    if "price" in df.columns:
        df["price_clean"] = df["price"].apply(_parse_price)
        df.loc[df["price_clean"] > 1e9, "price_clean"] = np.nan
        df["log_price"] = np.log1p(df["price_clean"].clip(lower=0))
    else:
        df["price_clean"] = np.nan
        df["log_price"] = np.nan

    for prefix, col, cats in [
        ("season_", "season", season_cats),
        ("room_", "room", room_cats),
        ("comp_", "exhibition_with", comp_cats),
    ]:
        if col in df.columns:
            bindf = _binarize_multi_label(df[col], cats)
        else:
            bindf = pd.DataFrame(0, index=df.index, columns=cats)
        for c in cats:
            df[f"{prefix}{c}"] = bindf[c].values

    # ── impute numeric ──
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(numeric_medians.get(col, 0.0))

    # ── TF-IDF ──
    texts = []
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in text_cols if c in df.columns and pd.notna(row.get(c))]
        texts.append(" ".join(parts))

    X_text = _texts_to_tfidf(texts, vocab, idf)

    # ── assemble feature matrix ──
    X_num = df[numeric_cols].values.astype(np.float64)
    X_bin = df[binary_cols].values.astype(np.float64)
    X = np.hstack([X_num, X_bin, X_text])

    # ── standardize ──
    X = (X - feature_means) / feature_stds

    # ── GDA prediction ──
    n = X.shape[0]
    log_post = np.zeros((n, K))

    if use_qda:
        for k in range(K):
            diff = X - class_means[k]
            mahal = np.sum(diff @ pc_cov_invs[k] * diff, axis=1)
            log_post[:, k] = np.log(class_priors[k]) - 0.5 * mahal - 0.5 * pc_cov_logdets[k]
    else:
        for k in range(K):
            diff = X - class_means[k]
            mahal = np.sum(diff @ shared_cov_inv * diff, axis=1)
            log_post[:, k] = np.log(class_priors[k]) - 0.5 * mahal - 0.5 * shared_cov_logdet

    pred_indices = np.argmax(log_post, axis=1)
    return [class_names[i] for i in pred_indices]
