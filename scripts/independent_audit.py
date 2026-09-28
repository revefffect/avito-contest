"""Recalculate the frozen independent audit with direct model inference.

Rechecks 87 claims about raw sizes and IDs, split separation, hidden pairs,
full-truth Recall@50 and candidate coverage, per-query aggregates, frozen CSV
hashes and target-disjoint sensitivity. No project metric code is imported.
No training, model selection or feature rebuild is performed.

--solution-root must be the full original working solution directory with raw
and processed data, saved validation features, models and reports. The compact
quick-reproduction ZIP and reference_solution source snapshot lack those data.
Provide the datasets separately to repeat this full audit.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--solution-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "deliverables/independent_audit_rerun"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    protected = [
        ROOT / name
        for name in ["report", "data_dump", "evidence", "reference_solution", "mini"]
    ]
    if args.output_dir.resolve().is_relative_to(args.solution_root.resolve()) or any(
        args.output_dir.resolve().is_relative_to(path) for path in protected
    ):
        parser.error("Output must be outside source data and frozen reports")
    if args.threads < 1:
        parser.error("Threads must be positive")
    required = [
        "data/raw/train.parquet",
        "data/processed/eval_truth.parquet",
        "artifacts/features_hybrid.parquet",
        "deliverables/verification.json",
        "deliverables/avito_solution.zip",
    ]
    missing = [name for name in required if not (args.solution_root / name).is_file()]
    if missing:
        parser.error(
            "Full original working solution required. Missing: " + ", ".join(missing)
        )
    return args


def run(args):
    from datetime import datetime, timezone
    import csv
    import hashlib
    import json
    from pathlib import Path
    import re
    import time
    import unicodedata

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from catboost import CatBoost

    SOURCE = args.solution_root.resolve()
    OUTPUT = args.output_dir.resolve()
    START = time.monotonic()
    SEARCH_COLUMNS = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    ID_PATTERN = re.compile("[0-9a-f]{16}")
    TOKEN_PATTERN = re.compile("[a-zа-я0-9]+")
    checks = []
    sources = {}

    def sha256(path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def register(relative):
        path = SOURCE / relative
        sources[relative] = {
            "absolute_path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        return path

    def read_table(relative, **kwargs):
        return pd.read_parquet(SOURCE / relative, use_threads=False, **kwargs)

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    def normalize_reference(value):
        if not isinstance(value, str):
            return ""
        text = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
        return " ".join(TOKEN_PATTERN.findall(text))

    def context_keys(frame):
        values = frame[SEARCH_COLUMNS].fillna("").astype(str).copy()
        values["search_query"] = values["search_query"].map(normalize_reference)
        values["search_infm_params_text"] = values["search_infm_params_text"].map(
            normalize_reference
        )
        return values.agg("\x1f".join, axis=1)

    def id_profile(values, kind):
        strings = values.map(lambda value: isinstance(value, str))
        if kind == "item":
            valid = values.map(
                lambda value: isinstance(value, str)
                and ID_PATTERN.fullmatch(value) is not None
            )
        else:
            valid = values.map(
                lambda value: isinstance(value, str)
                and len(value) == 16
                and not any(char.isspace() for char in value)
            )
        return {
            "rows": len(values),
            "unique": int(values.nunique(dropna=False)),
            "nulls": int(values.isna().sum()),
            "non_strings": int((~strings).sum()),
            "invalid_format": int((~valid).sum()),
            "duplicate_rows": int(values.duplicated().sum()),
        }

    print("Reading source data and checking separation", flush=True)
    raw_train = read_table(
        "data/raw/train.parquet", columns=SEARCH_COLUMNS + ["item_id"]
    )
    benchmark_queries = read_table("data/raw/benchmark_queries.parquet")
    benchmark_items = read_table(
        "data/raw/benchmark_items.parquet", columns=["item_id"]
    )
    corpus = read_table("data/processed/corpus.parquet", columns=["item_id"])
    retrieval = read_table("data/processed/retrieval_train.parquet")
    queries = read_table("data/processed/eval_queries.parquet")
    queries["qid"] = np.arange(len(queries), dtype=np.int32)
    truth = read_table("data/processed/eval_truth.parquet")
    raw_keys = context_keys(raw_train)
    retrieval_keys = context_keys(retrieval)
    query_keys = context_keys(queries)
    raw_pairs = pd.DataFrame({"context_key": raw_keys, "item_id": raw_train.item_id})
    raw_sizes = {
        "train": {
            **id_profile(raw_train.item_id, "item"),
            "unique_normalized_texts": int(
                raw_train.search_query.map(normalize_reference).nunique()
            ),
            "unique_query_contexts": int(raw_keys.nunique()),
            "unique_context_item_pairs": int(len(raw_pairs.drop_duplicates())),
            "repeated_context_item_pair_rows": int(raw_pairs.duplicated().sum()),
        },
        "benchmark_queries": id_profile(benchmark_queries.query_id, "query"),
        "benchmark_items": id_profile(benchmark_items.item_id, "item"),
        "processed_corpus": id_profile(corpus.item_id, "item"),
        "retrieval_train_rows": len(retrieval),
        "extra_corpus_items": len(corpus) - len(benchmark_items),
    }
    raw_sizes["train"][
        "duplicate_rows_definition"
    ] = "Repeated occurrences of item_id, not duplicate full records"
    check(
        "source_row_counts",
        len(raw_train) == 497673
        and len(benchmark_queries) == 2452
        and len(benchmark_items) == 189212,
        actual={
            "train": len(raw_train),
            "benchmark_queries": len(benchmark_queries),
            "benchmark_items": len(benchmark_items),
        },
    )
    for name in ["benchmark_queries", "benchmark_items", "processed_corpus"]:
        stats = raw_sizes[name]
        check(
            name + "_ids_unique_and_valid",
            stats["duplicate_rows"] == stats["invalid_format"] == stats["nulls"] == 0,
        )
    check("train_item_ids_are_valid_strings", raw_sizes["train"]["invalid_format"] == 0)
    check(
        "corpus_benchmark_prefix_exact",
        corpus.item_id.iloc[: len(benchmark_items)].tolist()
        == benchmark_items.item_id.tolist(),
    )
    check(
        "retrieval_context_keys_match_independent_recomputation",
        retrieval_keys.equals(retrieval.context_key),
        mismatched_rows=int((retrieval_keys != retrieval.context_key).sum()),
    )
    check(
        "query_context_keys_match_independent_recomputation",
        query_keys.equals(queries.context_key),
        mismatched_rows=int((query_keys != queries.context_key).sum()),
    )
    retrieval_texts = retrieval.search_query.map(normalize_reference)
    check(
        "retrieval_text_keys_match_independent_recomputation",
        retrieval_texts.equals(retrieval.text_key),
    )
    check(
        "query_text_keys_match_independent_recomputation",
        queries.search_query.map(normalize_reference).equals(queries.text_key),
    )
    query_lookup = queries.set_index("query_id")
    hidden = truth.merge(
        queries[["query_id", "context_key"]], on="query_id", validate="many_to_one"
    )
    hidden_pairs = set(zip(hidden.context_key, hidden.item_id))
    retrieval_pairs = set(zip(retrieval_keys, retrieval.item_id))
    hidden_overlap = hidden_pairs & retrieval_pairs
    cold_texts = set(queries.loc[queries.regime.eq("cold_text"), "text_key"])
    cold_overlap = cold_texts & set(retrieval_texts)
    cold_items = set(truth.loc[~truth.seen_item.astype(bool), "item_id"])
    cold_item_overlap = cold_items & set(retrieval.item_id)
    check(
        "no_hidden_context_item_pairs_in_retrieval_train",
        not hidden_overlap,
        overlap_count=len(hidden_overlap),
    )
    check(
        "no_cold_texts_in_retrieval_train",
        not cold_overlap,
        overlap_count=len(cold_overlap),
    )
    check(
        "no_declared_unseen_target_items_in_retrieval_train",
        not cold_item_overlap,
        overlap_count=len(cold_item_overlap),
    )
    check(
        "truth_split_matches_query_split",
        (
            truth["split"].to_numpy()
            == truth.query_id.map(query_lookup["split"]).to_numpy()
        ).all(),
    )
    check("all_truth_items_are_in_corpus", truth.item_id.isin(corpus.item_id).all())
    check(
        "truth_seen_item_flag_matches_retrieval_train",
        np.array_equal(
            truth.seen_item.to_numpy(), truth.item_id.isin(retrieval.item_id).to_numpy()
        ),
    )
    separation = {
        "hidden_pair_count": len(hidden_pairs),
        "hidden_pair_overlap": len(hidden_overlap),
        "cold_text_count": len(cold_texts),
        "cold_text_overlap": len(cold_overlap),
        "declared_unseen_target_items": len(cold_items),
        "unseen_item_overlap": len(cold_item_overlap),
        "query_counts": queries.groupby("split").size().to_dict(),
        "split_intersections": {},
    }
    for first, second in [("fit", "dev"), ("fit", "audit"), ("dev", "audit")]:
        a = queries.loc[queries["split"].eq(first)]
        b = queries.loc[queries["split"].eq(second)]
        overlaps = {
            column: len(set(a[column]) & set(b[column]))
            for column in ["query_id", "context_key", "text_key"]
        }
        shared_targets = set(truth.loc[truth["split"].eq(first), "item_id"]) & set(
            truth.loc[truth["split"].eq(second), "item_id"]
        )
        separation["split_intersections"][f"{first}/{second}"] = {
            **overlaps,
            "shared_relevant_item_ids": len(shared_targets),
        }
        check(
            f"{first}_{second}_queries_contexts_texts_disjoint",
            not any(overlaps.values()),
            intersections=overlaps,
        )

    print("Recomputing fixed audit predictions without evaluation.py", flush=True)
    config = json.loads(register("configs/final.json").read_text())
    selection = json.loads(register("configs/selection.json").read_text())
    audit_queries = queries.loc[queries["split"].eq("audit")].copy()
    audit_truth = truth.loc[truth["split"].eq("audit")].copy()
    item_position = {item_id: index for index, item_id in enumerate(corpus.item_id)}
    relevant = {
        query_id: {item_position[item_id] for item_id in values}
        for query_id, values in audit_truth.groupby("query_id").item_id
    }
    feature_lists = {}
    required = {"qid", "item_idx", "label"}
    for model in config["models"]:
        names_path = str(Path(model["path"]).with_suffix(".feature_names.json"))
        names = json.loads(register(names_path).read_text())
        feature_lists[model["path"]] = names
        required.update(names)
        check(
            "model_excludes_ids_and_labels_" + Path(model["path"]).stem,
            not set(names)
            & {"qid", "item_idx", "query_id", "item_id", "label", "split", "target"},
        )
        saved = json.loads(
            register(
                str(Path(model["path"]).with_suffix(".train_config.json"))
            ).read_text()
        )
        check(
            "saved_training_split_" + Path(model["path"]).stem,
            set(saved["fit_splits"].split(",")) == {"fit", "dev"}
            and saved["fixed_iterations"]
            and saved["fit_query_count"] == 4500,
            fit_splits=saved["fit_splits"],
            fixed_iterations=saved["fixed_iterations"],
            fit_query_count=saved["fit_query_count"],
        )
    features_path = config["validation_features"]
    features = read_table(
        features_path,
        columns=sorted(required),
        filters=[("qid", "in", audit_queries.qid.tolist())],
    )
    check(
        "audit_candidate_qid_coverage_exact",
        set(features.qid) == set(audit_queries.qid),
    )
    check(
        "audit_candidates_unique", not features[["qid", "item_idx"]].duplicated().any()
    )
    check(
        "audit_candidate_indices_in_range",
        features.item_idx.between(0, len(corpus) - 1).all(),
    )
    positions = features.groupby("qid", sort=True).indices
    scores = {}
    model_training_metadata = []
    for model_info in config["models"]:
        path = register(model_info["path"])
        model = CatBoost()
        model.load_model(path)
        names = feature_lists[model_info["path"]]
        check("model_feature_order_" + path.stem, model.feature_names_ == names)
        prediction = np.asarray(
            model.predict(
                features[names],
                prediction_type="RawFormulaVal",
                thread_count=args.threads,
            ),
            dtype=np.float64,
        ).reshape(-1)
        check("finite_model_scores_" + path.stem, np.isfinite(prediction).all())
        scores[path.stem] = prediction
        model_training_metadata.append(
            {
                "path": model_info["path"],
                "trees": model.tree_count_,
                "feature_count": len(names),
            }
        )
    normalized = {}
    for name, values in scores.items():
        transformed = np.zeros(len(values), dtype=np.float64)
        for rows in positions.values():
            local = values[rows]
            std = local.std(ddof=0)
            if std > 0:
                transformed[rows] = (local - local.mean()) / std
        normalized[name] = transformed
    weights = np.asarray(config["weights"], dtype=np.float64)
    weights /= weights.sum()
    blend_name = "blend_" + "_".join(f"{value:g}" for value in weights)
    scores[blend_name] = sum(
        weight * normalized[name] for weight, name in zip(weights, normalized)
    )
    independent_rows = []
    missing_total = 0
    pool_queries_missing_any = 0
    pool_queries_missing_all = 0
    label_disagreements = 0
    for query in audit_queries.to_dict("records"):
        qid, query_id = query["qid"], query["query_id"]
        rows = positions.get(qid, np.empty(0, dtype=np.int64))
        items = features.item_idx.to_numpy()[rows]
        true_items = relevant[query_id]
        pool = set(map(int, items))
        missing = len(true_items - pool)
        missing_total += missing
        pool_queries_missing_any += int(missing > 0)
        pool_queries_missing_all += int(missing == len(true_items))
        direct_labels = np.fromiter(
            (int(item in true_items) for item in items), dtype=np.int8
        )
        label_disagreements += int(
            np.count_nonzero(direct_labels != features.label.to_numpy()[rows])
        )
        filters = query["search_infm_params_text"]
        empty = not isinstance(filters, str) or not filters.strip()
        for model_name, predictions in scores.items():
            order = np.lexsort((items, -predictions[rows]))
            ordered_items = items[order]
            record = {
                "qid": qid,
                "query_id": query_id,
                "model_name": model_name,
                "relevant_count": len(true_items),
                "candidate_count": len(pool),
                "filter_empty": empty,
                "recall@union": len(true_items & pool) / len(true_items),
            }
            for k in [10, 50, 100]:
                record[f"recall@{k}"] = len(
                    true_items & set(map(int, ordered_items[:k]))
                ) / len(true_items)
            independent_rows.append(record)
    independent = pd.DataFrame(independent_rows)
    check(
        "candidate_labels_match_independent_truth_lookup",
        label_disagreements == 0,
        disagreeing_rows=label_disagreements,
    )
    reference_fraction = float(
        benchmark_queries.search_infm_params_text.map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).mean()
    )
    metrics = {}
    for name, rows in independent.groupby("model_name", sort=False):
        result = {
            metric: float(rows[metric].mean())
            for metric in ["recall@10", "recall@50", "recall@100", "recall@union"]
        }
        group_means = rows.groupby("filter_empty")["recall@50"].mean()
        result["poststratified_recall@50"] = float(
            reference_fraction * group_means.loc[True]
            + (1 - reference_fraction) * group_means.loc[False]
        )
        result["filter_groups"] = {
            str(bool(key)): {
                "queries": len(group),
                "recall@50": float(group["recall@50"].mean()),
                "recall@union": float(group["recall@union"].mean()),
            }
            for key, group in rows.groupby("filter_empty")
        }
        metrics[name] = result
    saved_report = json.loads(register("reports/final_audit.json").read_text())
    saved_per_query = read_table("reports/final_audit.per_query.parquet")
    register("reports/final_audit.per_query.parquet")
    joined = independent.merge(
        saved_per_query,
        on=["qid", "query_id", "model_name"],
        suffixes=("_independent", "_saved"),
        validate="one_to_one",
    )
    check(
        "saved_per_query_exact_coverage",
        len(joined) == len(independent) == len(saved_per_query),
    )
    per_query_differences = {}
    for column in [
        "relevant_count",
        "candidate_count",
        "recall@10",
        "recall@50",
        "recall@100",
        "recall@union",
    ]:
        delta = np.abs(joined[column + "_independent"] - joined[column + "_saved"])
        maximum = float(delta.max())
        per_query_differences[column] = maximum
        check(
            "saved_per_query_" + column,
            maximum < 1e-12,
            max_absolute_difference=maximum,
        )
    check(
        "saved_per_query_filter_flags",
        (joined.filter_empty_independent == joined.filter_empty_saved).all(),
    )
    for name, result in metrics.items():
        for key in [
            "recall@10",
            "recall@50",
            "recall@100",
            "recall@union",
            "poststratified_recall@50",
        ]:
            difference = abs(result[key] - saved_report["results"][name][key])
            check(
                "saved_aggregate_" + name + "_" + key,
                difference < 1e-12,
                absolute_difference=difference,
            )
    check(
        "saved_reference_filter_fraction",
        abs(reference_fraction - saved_report["reference_empty_fraction"]) < 1e-15,
    )
    final_results = json.loads(register("reports/final_results.json").read_text())
    check(
        "saved_primary_result",
        abs(
            metrics[blend_name]["recall@50"]
            - final_results["primary_audit"]["recall@50"]
        )
        < 1e-12,
    )
    check(
        "saved_selection_unchanged_after_audit",
        final_results["selection_changed_after_audit"] is False,
    )
    check(
        "frozen_model_recipe_matches_selection",
        config["models"] == selection["models"]
        and config["weights"] == selection["weights"],
    )
    primary = independent.loc[independent.model_name.eq(blend_name)]
    wrong_denominator = []
    for query in audit_queries.to_dict("records"):
        rows = positions.get(query["qid"], np.empty(0, dtype=np.int64))
        found_count = len(
            relevant[query["query_id"]]
            & set(map(int, features.item_idx.to_numpy()[rows]))
        )
        if found_count:
            correct = primary.loc[primary.qid.eq(query["qid"]), "recall@50"].iloc[0]
            hits = correct * len(relevant[query["query_id"]])
            wrong_denominator.append(hits / found_count)
    lexical = read_table(
        "artifacts/features_lexical.parquet",
        columns=["qid", "item_idx"],
        filters=[("qid", "in", audit_queries.qid.tolist())],
    )
    lex_groups = lexical.groupby("qid").item_idx.agg(set).to_dict()
    lexical_pool_recall = []
    lexical_missing = 0
    for query in audit_queries.to_dict("records"):
        relevant_set = relevant[query["query_id"]]
        pool = lex_groups.get(query["qid"], set())
        lexical_pool_recall.append(len(relevant_set & pool) / len(relevant_set))
        lexical_missing += len(relevant_set - pool)
    pool_report = {
        "audit_queries": len(audit_queries),
        "unique_truth_pairs": int(
            audit_truth[["query_id", "item_id"]].drop_duplicates().shape[0]
        ),
        "raw_truth_rows": len(audit_truth),
        "candidate_rows": len(features),
        "candidate_count_min": int(primary.candidate_count.min()),
        "candidate_count_max": int(primary.candidate_count.max()),
        "candidate_count_mean": float(primary.candidate_count.mean()),
        "hybrid_macro_recall_union": metrics[blend_name]["recall@union"],
        "relevant_items_missing_from_hybrid_pool": missing_total,
        "queries_with_any_missing_relevant": pool_queries_missing_any,
        "queries_with_all_relevant_missing": pool_queries_missing_all,
        "macro_recall50_with_full_truth_denominator": metrics[blend_name]["recall@50"],
        "invalid_recall50_if_using_retrieved_positive_denominator_and_dropping_zero_positive_queries": float(
            np.mean(wrong_denominator)
        ),
        "lexical_pool_same_audit": {
            "candidate_rows": len(lexical),
            "macro_recall_union": float(np.mean(lexical_pool_recall)),
            "missing_relevant_items": lexical_missing,
        },
        "lexical_comparison_role": "Retrospective fixed-pool coverage check only, not model selection",
    }

    print(
        "Cross-checking three saved CSV files and prior archive verification",
        flush=True,
    )
    verification = json.loads(register("deliverables/verification.json").read_text())
    verification_by_config = {
        record["config"]: record for record in verification["reproductions"]
    }
    answers = {}
    query_ids = benchmark_queries.query_id.tolist()
    allowed_ids = set(benchmark_items.item_id)
    for config_name, csv_name in [
        ("configs/final.json", "answer.csv"),
        ("configs/with_history.json", "answer_with_history.csv"),
        ("configs/no_history.json", "answer_no_history.csv"),
    ]:
        cfg = json.loads(register(config_name).read_text())
        path = register(csv_name)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            records = list(reader)
        ids = [row["query_id"] for row in records]
        bad_rows = 0
        counts = []
        for row in records:
            values = row["answer"].split(" ") if row["answer"] else []
            counts.append(len(values))
            bad_rows += int(
                len(values) > 50
                or len(values) != len(set(values))
                or any(
                    ID_PATTERN.fullmatch(value) is None or value not in allowed_ids
                    for value in values
                )
            )
        digest = sha256(path)
        prior = verification_by_config[config_name]
        equal = (
            digest
            == cfg["answer_sha256"]
            == prior["csv_sha256"]
            == prior["expected_csv_sha256"]
        )
        valid = (
            columns == ["query_id", "answer"]
            and len(ids) == len(set(ids))
            and set(ids) == set(query_ids)
            and bad_rows == 0
        )
        check("csv_schema_ids_and_membership_" + csv_name, valid, invalid_rows=bad_rows)
        check("csv_config_prior_verification_hash_" + csv_name, equal)
        answers[csv_name] = {
            "sha256": digest,
            "config": config_name,
            "query_rows": len(records),
            "candidates_min": min(counts),
            "candidates_max": max(counts),
            "valid": valid,
            "matches_config_and_prior_regeneration": equal,
        }
        for relative, expected_hash in cfg["input_sha256"].items():
            if relative not in sources:
                register(relative)
            check(
                "frozen_input_hash_" + config_name + "_" + relative,
                sources[relative]["sha256"] == expected_hash,
            )
    archive_hash = sha256(SOURCE / "deliverables/avito_solution.zip")
    check(
        "archive_hash_matches_prior_verification",
        archive_hash == verification["archive_sha256"],
    )
    check(
        "prior_regeneration_report_passed",
        verification["status"] == "passed"
        and verification["manifest"]["passed"]
        and verification["all_regenerated_outputs_match"],
    )
    for relative in [
        "data/raw/train.parquet",
        "data/raw/benchmark_queries.parquet",
        "data/raw/benchmark_items.parquet",
        "data/processed/retrieval_train.parquet",
        "data/processed/eval_queries.parquet",
        "data/processed/eval_truth.parquet",
        "data/processed/corpus.parquet",
        features_path,
        "artifacts/features_lexical.parquet",
    ]:
        register(relative)

    training_targets = set(truth.loc[truth["split"].isin(["fit", "dev"]), "item_id"])
    shared_targets = set(audit_truth.item_id) & training_targets
    removed_query_ids = set(
        audit_truth.loc[audit_truth.item_id.isin(training_targets), "query_id"]
    )
    retained = primary.loc[~primary.query_id.isin(removed_query_ids)]
    retained_truth = audit_truth.loc[~audit_truth.query_id.isin(removed_query_ids)]
    remaining_overlap = set(retained_truth.item_id) & training_targets
    sensitivity_groups = retained.groupby("filter_empty")["recall@50"].mean()
    sensitivity = {
        "definition": "Keep an audit query only if none of its target item IDs occurs in fit or dev truth.",
        "role": "Post hoc sensitivity of the frozen primary ensemble only. No model or weight selection.",
        "shared_target_item_ids": len(shared_targets),
        "removed_queries": len(removed_query_ids),
        "removed_positive_pairs": len(audit_truth) - len(retained_truth),
        "remaining_queries": len(retained),
        "remaining_positive_pairs": len(retained_truth),
        "remaining_target_item_overlap": len(remaining_overlap),
        "macro_recall50": float(retained["recall@50"].mean()),
        "poststratified_recall50": float(
            reference_fraction * sensitivity_groups.loc[True]
            + (1 - reference_fraction) * sensitivity_groups.loc[False]
        ),
        "reference_empty_filter_fraction": reference_fraction,
        "filter_groups": {
            str(bool(key)): {
                "queries": len(group),
                "macro_recall50": float(group["recall@50"].mean()),
            }
            for key, group in retained.groupby("filter_empty")
        },
        "method": "Subset independently reconstructed per-query results using an exclusion mask computed from full truth target item IDs.",
        "interpretation_limit": "The target exclusion does not make the shared retrieval corpus item-disjoint. Covariate overlap and unlabeled relevance remain possible.",
    }
    separation["final_fit_dev_vs_audit_shared_relevant_item_ids"] = len(shared_targets)
    check(
        "sensitivity_no_target_overlap_after_query_exclusion",
        not remaining_overlap,
        remaining_target_item_overlap=len(remaining_overlap),
    )

    unknowns = [
        "Разделение является query-disjoint, но не полностью item-disjoint. Целевые объявления пересекаются между fit, dev и audit.",
        "Скрытая метрика платформы не наблюдалась. Локальный audit не равен результату benchmark.",
        "Положительные пары отражают наблюдённый выбор. Полная релевантность остальных объявлений неизвестна.",
        "Перенос с учётом доли пустых фильтров корректирует только один наблюдаемый сдвиг состава запросов.",
        "Проверен один сохранённый split. Независимое повторное обучение и вариативность между split здесь не измерялись.",
        "Исторический порядок выбора модели подтверждается сохранёнными конфигурациями и отчётами, а не независимым журналом событий.",
        "Повторное получение трёх CSV в этой проверке не запускалось. Проверены текущие хеши и прежний успешный отчёт фактического воспроизведения.",
    ]
    report = {
        "status": (
            "passed" if all(value["passed"] for value in checks) else "failed_checks"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only independent verification of frozen data, models, audit metrics and submissions. No training, tuning or selection.",
        "independence": {
            "imported_project_modules": [],
            "metric_implementation": "Direct set intersections and arithmetic means from full truth",
            "ensemble_implementation": "NumPy per-query mean and population standard deviation, then frozen weighted sum",
            "ranking": "Descending score with ascending corpus row index for ties",
        },
        "raw_data": raw_sizes,
        "separation": separation,
        "audit": {
            "models": model_training_metadata,
            "frozen_weights": weights.tolist(),
            "primary_name": blend_name,
            "metrics": metrics,
            "reference_empty_filter_fraction": reference_fraction,
            "per_query_max_absolute_differences": per_query_differences,
            "candidate_pool": pool_report,
            "target_disjoint_sensitivity": sensitivity,
        },
        "submissions": answers,
        "archive_sha256": archive_hash,
        "checks": checks,
        "checks_passed": sum(value["passed"] for value in checks),
        "checks_total": len(checks),
        "unknowns_and_limits": unknowns,
        "sources": sources,
        "elapsed_seconds": round(time.monotonic() - START, 3),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "independent_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    primary_metrics = metrics[blend_name]
    failures = [value["name"] for value in checks if not value["passed"]]
    from urllib.parse import quote
    import os

    source_link = quote(os.path.relpath(SOURCE, OUTPUT).replace(os.sep, "/"))
    text = f"""# Независимая проверка сохранённого решения

    Проверка выполнена без обучения, подбора весов и изменения исходного решения. Исходная папка `avito contest` использовалась только для чтения. Итог: {report['checks_passed']} из {report['checks_total']} проверок пройдены.

    ## Данные и разделение

    В исходных файлах {len(raw_train):,} обучающих строк, {len(benchmark_queries):,} benchmark-запроса и {len(benchmark_items):,} объявления. Идентификаторы benchmark уникальны, имеют ожидаемый строковый формат и сохраняют исходный регистр. Расширенный корпус содержит {len(corpus):,} объявлений, из них {len(corpus)-len(benchmark_items):,} добавлены для локальной проверки.

    В `retrieval_train` отсутствуют все {len(hidden_pairs):,} скрытые пары «контекст запроса, объявление» и все {len(cold_texts):,} выбранные новые тексты. Пересечения равны нулю. Идентификаторы, контексты и нормализованные тексты запросов fit, dev и audit попарно не пересекаются. Размеры частей составляют {separation['query_counts']['fit']}, {separation['query_counts']['dev']} и {separation['query_counts']['audit']} запросов.

    ## Прямой пересчёт качества

    Сохранённые CatBoost-модели заново рассчитали оценки только для audit. Реализация метрики из проекта не импортировалась. Для каждого запроса объявления упорядочены по оценке с разрешением ничьих по позиции в корпусе. Числитель равен пересечению первых 50 кандидатов со всей доступной разметкой, знаменатель равен числу всех известных положительных объявлений этого запроса. Затем взято обычное среднее по {len(audit_queries)} запросам.

    Для зафиксированного ансамбля Recall@50 равен **{primary_metrics['recall@50']:.12f}**. После взвешивания групп по доле пустых фильтров в benchmark он равен **{primary_metrics['poststratified_recall@50']:.12f}**. Доля пустых фильтров независимо пересчитана как {reference_fraction:.12f}. Все сохранённые значения по запросам и итоговые метрики трёх фиксированных вариантов совпали с пересчётом с допуском 1e-12.

    Полнота исходного гибридного пула равна {primary_metrics['recall@union']:.12f}. В нём не хватает {missing_total} положительных объявлений у {pool_queries_missing_any} запросов, из которых {pool_queries_missing_all} не имеют ни одного найденного положительного объявления. Пропущенные положительные объявления остались в знаменателе. Ошибочная замена знаменателя на найденные положительные с удалением нулевых запросов дала бы {pool_report['invalid_recall50_if_using_retrieved_positive_denominator_and_dropping_zero_positive_queries']:.12f} и завысила бы оценку.

    Для описательной сверки также измерена полнота ранее сохранённого лексического пула на тех же audit-запросах: {pool_report['lexical_pool_same_audit']['macro_recall_union']:.12f}. Значение не использовалось для выбора модели.

    ## Чувствительность к общим целевым объявлениям

    Разделение запросов не гарантирует разделение целевых объявлений. Между fit+dev и audit найдены {len(shared_targets)} общих item_id. После удаления {len(removed_query_ids)} audit-запросов, у которых хотя бы одна цель встречалась в fit+dev, осталось {len(retained)} запросов и {len(retained_truth)} положительных пар. Для них Recall@50 равен {sensitivity['macro_recall50']:.12f}, взвешенная оценка равна {sensitivity['poststratified_recall50']:.12f}. Повторного обучения или выбора модели не было. Общий корпус кандидатов остаётся общим.

    ## Файлы ответа

    Три текущих CSV имеют по {len(benchmark_queries)} уникальных запросов и по 50 разрешённых объявлений в строке. Хеши совпадают одновременно с конфигурациями и с прежним отчётом фактического воспроизведения из ZIP. Сам ZIP также совпал с ранее проверенным архивом. Архив в этой проверке заново не распаковывался.

    ## Неизвестности и границы

    """
    text += "\n".join("- " + value for value in unknowns) + "\n\n"
    text += """## Источники и метод

    Полные значения, результаты каждой проверки, SHA256 исходных файлов и абсолютные пути сохранены в [independent_audit.json](independent_audit.json).

    - Исходные размеры и ID: [train.parquet](../../avito%20contest/data/raw/train.parquet), [benchmark_queries.parquet](../../avito%20contest/data/raw/benchmark_queries.parquet), [benchmark_items.parquet](../../avito%20contest/data/raw/benchmark_items.parquet).
    - Проверка разделения: [retrieval_train.parquet](../../avito%20contest/data/processed/retrieval_train.parquet), [eval_queries.parquet](../../avito%20contest/data/processed/eval_queries.parquet), [eval_truth.parquet](../../avito%20contest/data/processed/eval_truth.parquet).
    - Прямой пересчёт: [final.json](../../avito%20contest/configs/final.json), [features_hybrid.parquet](../../avito%20contest/artifacts/features_hybrid.parquet), [сохранённые показатели по запросам](../../avito%20contest/reports/final_audit.per_query.parquet).
    - Воспроизведение ответов: [verification.json](../../avito%20contest/deliverables/verification.json).
    """
    text = text.replace("../../avito%20contest", source_link)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", " ", text)
    if failures:
        text += (
            "\n## Непройденные проверки\n\n"
            + "\n".join("- " + name for name in failures)
            + "\n"
        )
    text = re.sub(r"(?m)^    ", "", text)
    assert "\u2014" not in text and ";" not in text
    (OUTPUT / "independent_audit.md").write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks_passed": report["checks_passed"],
                "checks_total": report["checks_total"],
                "primary": primary_metrics,
                "pool": pool_report,
                "failed": failures,
                "output": str(OUTPUT),
                "seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run(parse_arguments())
