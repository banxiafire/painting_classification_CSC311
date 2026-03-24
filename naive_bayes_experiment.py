import json
import re
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, MultiLabelBinarizer

DATA_PATH = Path(__file__).resolve().parent / "training_data_202601.csv"
TARGET_COL = "Painting"
GROUP_COL = "unique_id"
LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]
NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?",
]
MULTI_LABEL_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]
TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]


def extract_rating(value):
    if pd.isna(value):
        return np.nan
    match = re.match(r"^(\d+)", str(value).strip())
    if match is None:
        return np.nan
    return float(match.group(1))


def parse_price(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().lower().replace(",", "")

    range_match = re.search(r"(\d+\.?\d*)\s*(?:-|–|to)\s*(\d+\.?\d*)", cleaned)
    if range_match is not None:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0

    for pattern, multiplier in [
        (r"(\d+\.?\d*)\s*(?:billion|bn|b)\b", 1_000_000_000),
        (r"(\d+\.?\d*)\s*(?:million|m)\b", 1_000_000),
        (r"(\d+\.?\d*)\s*k\b", 1_000),
    ]:
        match = re.search(pattern, cleaned)
        if match is not None:
            return float(match.group(1)) * multiplier

    cleaned = cleaned.replace("canadian", "").replace("dollars", "")
    cleaned = cleaned.replace("dollar", "").replace("cad", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if match is None:
        return np.nan
    return float(match.group(0))


def split_multilabel(value):
    if pd.isna(value):
        return []
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]


def prepare_dataframe(df):
    prepared = df.copy()
    for col in LIKERT_COLS:
        prepared[col] = prepared[col].apply(extract_rating)
    prepared["How much (in Canadian dollars) would you be willing to pay for this painting?"] = prepared[
        "How much (in Canadian dollars) would you be willing to pay for this painting?"
    ].apply(parse_price)
    return prepared


def combine_text(df):
    text_parts = [df[col].fillna("").astype(str) for col in TEXT_COLS]
    combined = text_parts[0]
    for part in text_parts[1:]:
        combined = combined.str.cat(part, sep=" ")
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def preprocess_numeric_dataframe(df, price_cap=None):
    numeric_df = df[NUMERIC_COLS + LIKERT_COLS].apply(pd.to_numeric, errors="coerce").copy()
    price_col = "How much (in Canadian dollars) would you be willing to pay for this painting?"
    if price_cap is None:
        price_cap = numeric_df[price_col].quantile(0.99)
    if pd.isna(price_cap):
        price_cap = 0.0
    numeric_df[price_col] = numeric_df[price_col].clip(lower=0, upper=float(price_cap))
    numeric_df[price_col] = np.log1p(numeric_df[price_col])
    return numeric_df, float(price_cap)


def maybe_binarize_matrix(matrix):
    binary_matrix = matrix.tocsr(copy=True)
    if binary_matrix.nnz > 0:
        binary_matrix.data = np.ones_like(binary_matrix.data)
    return binary_matrix


def fit_feature_bundle(train_df, config):
    numeric_df, price_cap = preprocess_numeric_dataframe(train_df)
    numeric_fill = numeric_df.median().fillna(0.0)
    numeric_df = numeric_df.fillna(numeric_fill)

    if config["use_numeric"]:
        numeric_binner = KBinsDiscretizer(
            n_bins=config["numeric_bins"],
            encode="onehot",
            strategy="quantile",
        )
        numeric_matrix = numeric_binner.fit_transform(numeric_df)
    else:
        numeric_binner = None
        numeric_matrix = None

    if config["text_representation"] == "count":
        text_vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=config["ngram_range"],
            min_df=config["min_df"],
            max_features=config["max_features"],
            binary=config["binary_text"],
        )
    else:
        text_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=config["ngram_range"],
            min_df=config["min_df"],
            max_features=config["max_features"],
            sublinear_tf=True,
        )
    text_matrix = text_vectorizer.fit_transform(combine_text(train_df))

    multilabel_binarizers = {}
    multilabel_matrices = []
    if config["use_multilabel"]:
        for col in MULTI_LABEL_COLS:
            mlb = MultiLabelBinarizer(sparse_output=True)
            matrix = mlb.fit_transform(train_df[col].apply(split_multilabel))
            multilabel_binarizers[col] = mlb
            multilabel_matrices.append(matrix.tocsr())

    transformers = {
        "numeric_fill": numeric_fill,
        "numeric_binner": numeric_binner,
        "text_vectorizer": text_vectorizer,
        "multilabel_binarizers": multilabel_binarizers,
        "price_cap": price_cap,
        "model_type": config["model_type"],
    }

    feature_parts = [text_matrix]
    if numeric_matrix is not None:
        feature_parts.append(numeric_matrix)
    feature_parts.extend(multilabel_matrices)
    feature_matrix = hstack(feature_parts, format="csr")

    if config["model_type"] == "bernoulli":
        feature_matrix = maybe_binarize_matrix(feature_matrix)

    return transformers, feature_matrix


def transform_features(df, transformers):
    numeric_df, _ = preprocess_numeric_dataframe(df, price_cap=transformers["price_cap"])
    numeric_df = numeric_df.fillna(transformers["numeric_fill"])
    numeric_matrix = None
    if transformers["numeric_binner"] is not None:
        numeric_matrix = transformers["numeric_binner"].transform(numeric_df)

    text_matrix = transformers["text_vectorizer"].transform(combine_text(df))

    multilabel_matrices = []
    for col in MULTI_LABEL_COLS:
        if col not in transformers["multilabel_binarizers"]:
            continue
        mlb = transformers["multilabel_binarizers"][col]
        matrix = mlb.transform(df[col].apply(split_multilabel))
        multilabel_matrices.append(matrix.tocsr())

    feature_parts = [text_matrix]
    if numeric_matrix is not None:
        feature_parts.append(numeric_matrix)
    feature_parts.extend(multilabel_matrices)
    feature_matrix = hstack(feature_parts, format="csr")

    if isinstance(transformers["model_type"], str) and transformers["model_type"] == "bernoulli":
        feature_matrix = maybe_binarize_matrix(feature_matrix)

    return feature_matrix


def build_model(config):
    if config["model_type"] == "bernoulli":
        return BernoulliNB(alpha=config["alpha"], binarize=None)
    if config["model_type"] == "multinomial":
        return MultinomialNB(alpha=config["alpha"])
    return ComplementNB(alpha=config["alpha"])


def sanitize_probabilities(y_prob):
    cleaned = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    row_sums = cleaned.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0
    if np.any(zero_rows):
        cleaned[zero_rows] = 1.0 / cleaned.shape[1]
        row_sums = cleaned.sum(axis=1, keepdims=True)
    return cleaned / row_sums


def evaluate_predictions(y_true, y_pred, y_prob, labels):
    safe_prob = sanitize_probabilities(y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": float(log_loss(y_true, safe_prob, labels=labels)),
    }


def cross_validate(train_df, y_encoded, groups, label_values, config, n_splits=5):
    unique_group_count = pd.Series(groups).nunique()
    split_count = min(n_splits, int(unique_group_count))
    if split_count < 2:
        raise ValueError("Need at least two unique groups for grouped cross-validation.")

    cv = GroupKFold(n_splits=split_count)
    fold_metrics = []

    for fold_id, (fit_idx, val_idx) in enumerate(cv.split(train_df, y_encoded, groups), start=1):
        fold_train_df = train_df.iloc[fit_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_fit = y_encoded[fit_idx]
        y_val = y_encoded[val_idx]

        transformers, x_fit = fit_feature_bundle(fold_train_df, config)
        x_val = transform_features(fold_val_df, transformers)

        model = build_model(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model.fit(x_fit, y_fit)
            y_pred = model.predict(x_val)
            y_prob = sanitize_probabilities(model.predict_proba(x_val))
        metrics = evaluate_predictions(y_val, y_pred, y_prob, label_values)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    summary = {
        "mean_accuracy": float(np.mean([item["accuracy"] for item in fold_metrics])),
        "mean_macro_f1": float(np.mean([item["macro_f1"] for item in fold_metrics])),
        "mean_log_loss": float(np.mean([item["log_loss"] for item in fold_metrics])),
    }
    return fold_metrics, summary


def train_final_model(train_df, test_df, label_encoder, config):
    transformers, x_train = fit_feature_bundle(train_df, config)
    x_test = transform_features(test_df, transformers)

    y_train = label_encoder.transform(train_df[TARGET_COL])
    y_test = label_encoder.transform(test_df[TARGET_COL])

    model = build_model(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        train_prob = sanitize_probabilities(model.predict_proba(x_train))
        test_pred = model.predict(x_test)
        test_prob = sanitize_probabilities(model.predict_proba(x_test))

    train_metrics = evaluate_predictions(y_train, train_pred, train_prob, model.classes_)
    test_metrics = evaluate_predictions(y_test, test_pred, test_prob, model.classes_)
    test_confusion = confusion_matrix(y_test, test_pred, labels=model.classes_)

    prediction_table = test_df[[GROUP_COL, TARGET_COL]].copy()
    prediction_table["prediction"] = label_encoder.inverse_transform(test_pred)
    prediction_table["correct"] = prediction_table[TARGET_COL] == prediction_table["prediction"]

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": test_confusion.tolist(),
        "labels": label_encoder.inverse_transform(model.classes_).tolist(),
        "predictions": prediction_table,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = prepare_dataframe(df)
    df = df.dropna(subset=[TARGET_COL, GROUP_COL]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_COL])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_train_encoded = label_encoder.transform(train_df[TARGET_COL])
    train_groups = train_df[GROUP_COL].to_numpy()

    grid = list(
        product(
            ["multinomial", "complement", "bernoulli"],
            [0.1, 0.5, 1.0],
            ["count", "tfidf"],
            [(1, 1), (1, 2)],
            [1, 2],
            [800, 1600],
            [False, True],
            [True],
            [5],
        )
    )

    results = []
    for model_type, alpha, text_representation, ngram_range, min_df, max_features, use_numeric, use_multilabel, numeric_bins in grid:
        config = {
            "model_type": model_type,
            "alpha": alpha,
            "text_representation": text_representation,
            "binary_text": model_type == "bernoulli",
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_features": max_features,
            "use_numeric": use_numeric,
            "use_multilabel": use_multilabel,
            "numeric_bins": numeric_bins,
        }
        try:
            fold_metrics, summary = cross_validate(
                train_df=train_df,
                y_encoded=y_train_encoded,
                groups=train_groups,
                label_values=np.arange(len(label_encoder.classes_)),
                config=config,
                n_splits=5,
            )
        except ValueError:
            continue
        results.append({"config": config, "fold_metrics": fold_metrics, "summary": summary})

    if not results:
        raise RuntimeError("No Naive Bayes configuration completed successfully.")

    results.sort(
        key=lambda item: (
            -item["summary"]["mean_macro_f1"],
            -item["summary"]["mean_accuracy"],
            item["summary"]["mean_log_loss"],
        )
    )
    best_result = results[0]
    final_result = train_final_model(train_df, test_df, label_encoder, best_result["config"])

    printable = {
        "data": {
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_unique_id": int(train_df[GROUP_COL].nunique()),
            "test_unique_id": int(test_df[GROUP_COL].nunique()),
        },
        "best_config": {
            **best_result["config"],
            "ngram_range": list(best_result["config"]["ngram_range"]),
        },
        "cv_summary": best_result["summary"],
        "train_metrics": final_result["train_metrics"],
        "test_metrics": final_result["test_metrics"],
        "confusion_matrix": final_result["confusion_matrix"],
        "labels": final_result["labels"],
    }

    print(json.dumps(printable, indent=2))

    prediction_path = Path(__file__).resolve().parent / "naive_bayes_holdout_predictions.csv"
    final_result["predictions"].to_csv(prediction_path, index=False)
    print(f"Saved holdout predictions to: {prediction_path}")


if __name__ == "__main__":
    main()
