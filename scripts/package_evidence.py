"""Подготовить индекс доказательств и по отдельной команде собрать проверяемый архив.

Исходное решение читается из соседней папки avito contest. Запись разрешена
только в отдельную папку avito submission. По умолчанию ZIP не создаётся.
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "avito contest"

# Номера назначены явно. Новые источники добавляются в конец без перенумерации.
CATALOG = [
    ("E001", "Условие конкурса", "docs/task.md", "specification"),
    (
        "E002",
        "Происхождение и контрольные суммы набора данных",
        "provenance/dataset.json",
        "provenance",
    ),
    ("E003", "Профиль исходных данных", "reports/data_profile.json", "measurement"),
    ("E004", "География и фильтры", "reports/geography_analysis.json", "measurement"),
    ("E005", "Сдвиг категорий поиска", "reports/category_shift.json", "measurement"),
    (
        "E006",
        "Локальное разделение данных",
        "reports/validation_split.json",
        "measurement",
    ),
    (
        "E007",
        "Повторное построение разделения",
        "reports/split_reproducibility.json",
        "verification",
    ),
    (
        "E008",
        "Лексическая эвристика на dev",
        "reports/lexical_heuristic_dev.json",
        "measurement",
    ),
    (
        "E009",
        "Лексическая модель Logloss на dev",
        "reports/lexical_logloss_dev.json",
        "measurement",
    ),
    (
        "E010",
        "Сравнение лексических моделей",
        "reports/lexical_models_dev.json",
        "measurement",
    ),
    (
        "E011",
        "Гибридная эвристика на dev",
        "reports/hybrid_heuristic_dev.json",
        "measurement",
    ),
    (
        "E012",
        "Сравнение гибридных моделей",
        "reports/hybrid_models_dev.json",
        "measurement",
    ),
    (
        "E013",
        "Исключение прямой истории выбора",
        "reports/history_ablation_dev.json",
        "measurement",
    ),
    ("E014", "Протокол выбора до audit", "configs/selection.json", "protocol"),
    (
        "E015",
        "Результат независимой локальной проверки",
        "reports/final_audit.json",
        "measurement",
    ),
    (
        "E016",
        "Итоговые результаты и интервалы",
        "reports/final_results.json",
        "measurement",
    ),
    ("E017", "Окружение вычислений", "reports/runtime.json", "runtime"),
    (
        "E018",
        "Проверка пяти массивов E5",
        "reports/embedding_verification.json",
        "verification",
    ),
    (
        "E019",
        "Повторяемость признаков на трёх запросах",
        "reports/feature_reproducibility.json",
        "verification",
    ),
    (
        "E020",
        "Повторное получение трёх CSV из исходного архива",
        "verification.json",
        "verification",
    ),
    ("E021", "Финальная конфигурация решения", "configs/final.json", "configuration"),
    (
        "E022",
        "Контрольные суммы таблиц идентификаторов",
        "reports/portable_ids.json",
        "verification",
    ),
    (
        "E023",
        "Разбор ошибок лексического поиска",
        "reports/lexical_error_analysis.md",
        "analysis",
    ),
    ("E024", "Источник открытой модели E5", "provenance/model.json", "provenance"),
    (
        "E025",
        "Параметры обучения и кривые экспериментов",
        "provenance/training_runs.json",
        "inventory",
    ),
    (
        "E026",
        "Хеши данных, индексов, векторов, моделей и ответов",
        "provenance/artifacts.json",
        "inventory",
    ),
    (
        "E027",
        "Снимок исходного кода и его контрольные суммы",
        "provenance/source_snapshot.json",
        "inventory",
    ),
    ("E028", "Сохранённые журналы экспериментов", "provenance/logs.json", "inventory"),
    (
        "E029",
        "Независимый пересчёт средних метрик по таблицам запросов",
        "provenance/metric_recheck.json",
        "verification",
    ),
    (
        "E030",
        "Полная опись файлов data dump",
        "provenance/dump_inventory.json",
        "inventory",
    ),
    (
        "E031",
        "Опись исходного архива решения",
        "provenance/solution_manifest.json",
        "provenance",
    ),
    (
        "E032",
        "Независимая проверка метрик и отсутствия утечки",
        "independent/independent_audit.json",
        "verification",
    ),
    (
        "E033",
        "Фактический запуск отдельного мини-ноутбука",
        "independent/notebook_verification.json",
        "verification",
    ),
    (
        "E034",
        "Проверка основного benchmark-ответа",
        "reports/answer.report.json",
        "verification",
    ),
]

# Значения извлекаются из указанных JSON-полей, а не переписываются вручную.
CLAIMS = [
    (
        "E002",
        "/archive/sha256",
        "SHA256 исходного архива набора данных",
        "SHA256",
        "measurement",
    ),
    ("E002", "/files/train.parquet/rows", "Число строк train", "строк", "measurement"),
    (
        "E002",
        "/files/benchmark_queries.parquet/rows",
        "Число benchmark-запросов",
        "запросов",
        "measurement",
    ),
    (
        "E002",
        "/files/benchmark_items.parquet/rows",
        "Число объявлений benchmark",
        "объявлений",
        "measurement",
    ),
    (
        "E003",
        "/unique_train_items",
        "Число уникальных объявлений в train",
        "объявлений",
        "measurement",
    ),
    (
        "E003",
        "/unique_train_texts",
        "Число текстов train после lower, замены ё на е и свёртки пробелов",
        "текстов",
        "measurement",
    ),
    (
        "E003",
        "/benchmark_seen_text_fraction",
        "Доля benchmark-запросов со знакомым текстом",
        "доля",
        "measurement",
    ),
    (
        "E003",
        "/benchmark_seen_context_fraction",
        "Доля знакомых полных контекстов benchmark",
        "доля",
        "measurement",
    ),
    (
        "E003",
        "/corpus_items_seen_in_train",
        "Доля объявлений benchmark, встречавшихся в train",
        "доля корпуса",
        "measurement",
    ),
    (
        "E003",
        "/train_search_item_same_location",
        "Доля train-пар с совпадающей локацией",
        "доля пар",
        "measurement",
    ),
    (
        "E004",
        "/benchmark_query_fraction_with_exact_corpus_location",
        "Доля benchmark-запросов с объявлением той же локации в корпусе",
        "доля запросов",
        "measurement",
    ),
    (
        "E004",
        "/parsed_known_service_filter_pairs",
        "Число пар с распознанным видом услуги",
        "пар",
        "measurement",
    ),
    (
        "E004",
        "/parsed_service_pair_match_fraction",
        "Доля совпадений распознанного вида услуги",
        "доля пар",
        "measurement",
    ),
    (
        "E005",
        "/train_search_category_zero_rows",
        "Число train-строк с категорией поиска 0",
        "строк",
        "measurement",
    ),
    (
        "E005",
        "/benchmark_search_category_zero_rows",
        "Число benchmark-запросов с категорией поиска 0",
        "запросов",
        "measurement",
    ),
    ("E006", "/seed", "Seed локального разделения", "целое число", "design_choice"),
    (
        "E006",
        "/queries",
        "Общее число отложенных контекстов",
        "запросов",
        "measurement",
    ),
    (
        "E006",
        "/truth_pairs",
        "Число положительных отложенных пар",
        "пар",
        "measurement",
    ),
    ("E006", "/split_counts/fit", "Число запросов fit", "запросов", "measurement"),
    ("E006", "/split_counts/dev", "Число запросов dev", "запросов", "measurement"),
    ("E006", "/split_counts/audit", "Число запросов audit", "запросов", "measurement"),
    (
        "E006",
        "/validation_corpus_size",
        "Размер полного корпуса локальной проверки",
        "объявлений",
        "measurement",
    ),
    (
        "E006",
        "/extra_validation_items",
        "Число добавленных отложенных объявлений",
        "объявлений",
        "measurement",
    ),
    (
        "E006",
        "/seen_target_item_fraction",
        "Фактическая доля знакомых целевых объявлений после удаления истории",
        "доля целей",
        "measurement",
    ),
    (
        "E007",
        "/all_files_match",
        "Повторное построение дало те же четыре Parquet-файла по хешам",
        "логическое значение",
        "verification",
    ),
    (
        "E008",
        "/recall@50",
        "Recall@50 лексической эвристики на dev",
        "доля",
        "measurement",
    ),
    (
        "E008",
        "/recall@union",
        "Полнота объединения лексических кандидатов на dev",
        "доля",
        "measurement",
    ),
    (
        "E009",
        "/recall@50",
        "Recall@50 лексического Logloss на dev",
        "доля",
        "measurement",
    ),
    (
        "E010",
        "/results/ranker_lexical_softmax/recall@50",
        "Recall@50 лексического QuerySoftMax на dev",
        "доля",
        "measurement",
    ),
    (
        "E011",
        "/recall@50",
        "Recall@50 гибридной эвристики на dev",
        "доля",
        "measurement",
    ),
    (
        "E011",
        "/recall@union",
        "Полнота объединения гибридных кандидатов на dev",
        "доля",
        "measurement",
    ),
    (
        "E012",
        "/results/ranker_hybrid_d6/recall@50",
        "Recall@50 гибридной модели глубины 6 на dev",
        "доля",
        "measurement",
    ),
    (
        "E012",
        "/results/ranker_hybrid_d7/recall@50",
        "Recall@50 гибридной модели глубины 7 на dev",
        "доля",
        "measurement",
    ),
    (
        "E013",
        "/results/ranker_hybrid_no_history/recall@50",
        "Recall@50 модели без прямых счётчиков истории на dev",
        "доля",
        "measurement",
    ),
    (
        "E013",
        "/results/blend_0.5_0.5/recall@50",
        "Recall@50 выбранной равной смеси на dev",
        "доля",
        "measurement",
    ),
    (
        "E013",
        "/results/blend_0.5_0.5/poststratified_recall@50",
        "Recall@50 выбранной смеси с поправкой на долю пустых фильтров на dev",
        "доля",
        "measurement",
    ),
    (
        "E014",
        "/frozen_at_utc",
        "Время фиксации решения, записанное в протоколе выбора",
        "UTC",
        "reported_statement",
    ),
    (
        "E014",
        "/audit_quality_seen",
        "Флаг просмотра audit, записанный при выборе",
        "логическое значение",
        "reported_statement",
    ),
    (
        "E014",
        "/delta_vs_d6/recall@50/delta",
        "Разность dev Recall@50 между смесью и моделью глубины 6",
        "доля",
        "measurement",
    ),
    (
        "E014",
        "/delta_vs_d6/recall@50/ci95",
        "Парный bootstrap-интервал разности dev Recall@50",
        "95% интервал в долях",
        "measurement",
    ),
    (
        "E014",
        "/final_training_queries",
        "Число запросов финального обучения fit и dev",
        "запросов",
        "design_choice",
    ),
    (
        "E015",
        "/results/blend_0.5_0.5/recall@50",
        "Recall@50 выбранной смеси на независимом локальном audit",
        "доля",
        "measurement",
    ),
    (
        "E015",
        "/results/blend_0.5_0.5/poststratified_recall@50",
        "Recall@50 смеси на audit с поправкой на пустые фильтры",
        "доля",
        "measurement",
    ),
    (
        "E015",
        "/results/blend_0.5_0.5/recall@union",
        "Полнота объединённого набора на audit",
        "доля",
        "measurement",
    ),
    (
        "E015",
        "/results/final_d6/recall@50",
        "Recall@50 финальной модели с историей на audit",
        "доля",
        "measurement",
    ),
    (
        "E015",
        "/results/final_no_history/recall@50",
        "Recall@50 финальной модели без прямых счётчиков истории на audit",
        "доля",
        "measurement",
    ),
    (
        "E016",
        "/audit_recall50_ci95/blend_0.5_0.5",
        "Bootstrap-интервал audit Recall@50 выбранной смеси",
        "95% интервал в долях",
        "measurement",
    ),
    (
        "E016",
        "/bootstrap_samples",
        "Число bootstrap-выборок для итоговых интервалов",
        "выборок",
        "design_choice",
    ),
    (
        "E016",
        "/selection_changed_after_audit",
        "Флаг изменения выбора после audit в итоговом отчёте",
        "логическое значение",
        "reported_statement",
    ),
    (
        "E016",
        "/hidden_benchmark_score",
        "Скрытый результат платформы не измерен",
        "неизвестно",
        "reported_statement",
    ),
    (
        "E016",
        "/submission_attempts_used",
        "Число использованных попыток по итоговому отчёту",
        "попыток",
        "reported_statement",
    ),
    (
        "E016",
        "/answers/answer.csv/rows",
        "Число строк основного ответа",
        "строк",
        "measurement",
    ),
    (
        "E016",
        "/answers/answer.csv/min_items",
        "Минимальное число кандидатов в строке основного ответа",
        "объявлений",
        "measurement",
    ),
    (
        "E016",
        "/answers/answer.csv/max_items",
        "Максимальное число кандидатов в строке основного ответа",
        "объявлений",
        "measurement",
    ),
    (
        "E016",
        "/answers/answer.csv/sha256",
        "SHA256 основного CSV",
        "SHA256",
        "verification",
    ),
    (
        "E017",
        "/python",
        "Версия Python выполненных экспериментов",
        "версия",
        "measurement",
    ),
    (
        "E017",
        "/embedding_device",
        "Устройство расчёта сохранённых эмбеддингов",
        "устройство",
        "measurement",
    ),
    (
        "E017",
        "/embedding_precision",
        "Численная точность сохранённого расчёта E5",
        "тип",
        "measurement",
    ),
    (
        "E017",
        "/tests/passed",
        "Число тестов, указанное в сохранённом runtime-отчёте",
        "тестов",
        "reported_statement",
    ),
    (
        "E018",
        "/status",
        "Результат проверки пяти массивов E5",
        "статус",
        "verification",
    ),
    (
        "E018",
        "/model_revision",
        "Закреплённая ревизия модели E5",
        "git revision",
        "verification",
    ),
    (
        "E018",
        "/outputs/0/shape",
        "Форма матрицы эмбеддингов корпуса",
        "строки и размерность",
        "measurement",
    ),
    (
        "E018",
        "/outputs/0/max_norm_error",
        "Максимальное отклонение нормы векторов корпуса от единицы",
        "абсолютная ошибка",
        "verification",
    ),
    (
        "E019",
        "/compared_query_count",
        "Число запросов проверки повторяемости признаков",
        "запросов",
        "verification",
    ),
    (
        "E019",
        "/saved_rows",
        "Число строк признаков в проверке повторяемости",
        "строк",
        "verification",
    ),
    (
        "E019",
        "/max_absolute_difference",
        "Максимальная численная разность повторно построенных признаков",
        "абсолютная ошибка",
        "verification",
    ),
    (
        "E020",
        "/all_regenerated_outputs_match",
        "Все три повторно полученных CSV совпали с ожидаемыми хешами",
        "логическое значение",
        "verification",
    ),
    (
        "E020",
        "/archive_sha256",
        "SHA256 исходного solution-архива, проверенного в отдельной папке",
        "SHA256",
        "verification",
    ),
    (
        "E020",
        "/offline_execution/network_packet_capture_performed",
        "Сетевой packet capture при проверке не выполнялся",
        "логическое значение",
        "reported_statement",
    ),
    (
        "E021",
        "/weights",
        "Веса двух моделей в выбранной смеси",
        "веса",
        "design_choice",
    ),
    (
        "E021",
        "/dense/item_max_length",
        "Предельная длина входа объявления E5",
        "токенов",
        "design_choice",
    ),
    (
        "E021",
        "/dense/query_max_length",
        "Предельная длина входа запроса E5",
        "токенов",
        "design_choice",
    ),
    (
        "E024",
        "/revision",
        "Ревизия загруженной открытой модели по provenance",
        "git revision",
        "verification",
    ),
    (
        "E029",
        "/all_checks_passed",
        "Средние метрики совпали с независимым пересчётом таблиц запросов",
        "логическое значение",
        "verification",
    ),
    (
        "E032",
        "/checks_passed",
        "Число успешных проверок независимого аудита",
        "проверок",
        "verification",
    ),
    (
        "E032",
        "/checks_total",
        "Общее число проверок независимого аудита",
        "проверок",
        "verification",
    ),
    (
        "E032",
        "/raw_data/train/unique_normalized_texts",
        "Число текстов train после полного нормализатора NFKC, lower, ё и regex",
        "текстов",
        "measurement",
    ),
    (
        "E032",
        "/separation/hidden_pair_overlap",
        "Число скрытых положительных пар, оставшихся в retrieval_train",
        "пар",
        "verification",
    ),
    (
        "E032",
        "/separation/cold_text_overlap",
        "Число холодных текстов, оставшихся в retrieval_train",
        "текстов",
        "verification",
    ),
    (
        "E032",
        "/separation/final_fit_dev_vs_audit_shared_relevant_item_ids",
        "Общие целевые item_id между fit+dev и audit при разделении по запросам",
        "объявлений",
        "measurement",
    ),
    (
        "E032",
        "/audit/metrics/blend_0.5_0.5/recall@50",
        "Независимо пересчитанный Recall@50 выбранной смеси с полным знаменателем релевантности",
        "доля",
        "verification",
    ),
    (
        "E032",
        "/audit/candidate_pool/relevant_items_missing_from_hybrid_pool",
        "Положительные пары audit, отсутствующие среди гибридных кандидатов",
        "пар",
        "measurement",
    ),
    (
        "E032",
        "/audit/target_disjoint_sensitivity/removed_queries",
        "Исключённые audit-запросы с любым пересечением целей с fit или dev",
        "запросов",
        "measurement",
    ),
    (
        "E032",
        "/audit/target_disjoint_sensitivity/remaining_queries",
        "Число audit-запросов в проверке чувствительности без общих целей",
        "запросов",
        "measurement",
    ),
    (
        "E032",
        "/audit/target_disjoint_sensitivity/remaining_target_item_overlap",
        "Пересечение целевых объявлений после исключения запросов",
        "объявлений",
        "verification",
    ),
    (
        "E032",
        "/audit/target_disjoint_sensitivity/macro_recall50",
        "Recall@50 зафиксированной смеси без запросов с общими целями fit или dev",
        "доля",
        "measurement",
    ),
    (
        "E032",
        "/audit/target_disjoint_sensitivity/poststratified_recall50",
        "Recall@50 той же проверки чувствительности с поправкой на пустые фильтры",
        "доля",
        "measurement",
    ),
    (
        "E033",
        "/exact_csv_hash_match",
        "Новый мини-ноутбук независимо создал CSV с ожидаемым SHA256",
        "логическое значение",
        "verification",
    ),
    (
        "E033",
        "/ready_answer_present_before_execution",
        "Наличие готового answer.csv до независимого запуска мини-ноутбука",
        "логическое значение",
        "verification",
    ),
    (
        "E033",
        "/executed_code_cells",
        "Число фактически выполненных ячеек кода мини-ноутбука",
        "ячеек",
        "verification",
    ),
    (
        "E033",
        "/execution_seconds",
        "Длительность выполнения мини-ноутбука в записанном окружении",
        "секунд",
        "measurement",
    ),
    (
        "E033",
        "/generated_csv_sha256",
        "SHA256 CSV независимого запуска мини-ноутбука",
        "SHA256",
        "verification",
    ),
    (
        "E034",
        "/candidate_pool/rows",
        "Размер сохранённой benchmark-таблицы кандидатов",
        "пар запрос и объявление",
        "measurement",
    ),
    (
        "E034",
        "/submission/candidates_total",
        "Число кандидатов в основном конкурсном CSV",
        "пар запрос и объявление",
        "verification",
    ),
]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(f"Недопустимый относительный путь: {relative}")
    path = base / relative
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Символическая ссылка запрещена: {current}")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Путь вне выбранной папки: {relative}")
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def json_pointer(value, pointer):
    for part in pointer.strip("/").split("/") if pointer else []:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def claim_csv_rows(claims):
    """Вложенные значения сохраняем как JSON без потери типа и точности."""
    return [
        {
            key: (
                json.dumps(value, ensure_ascii=False)
                if key in {"value", "locator"}
                else value
            )
            for key, value in claim.items()
        }
        for claim in claims
    ]


def file_record(path, original_path=None):
    record = {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if original_path is not None:
        record["original_path"] = original_path
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        record.update(
            rows=parquet.metadata.num_rows, columns=parquet.schema_arrow.names
        )
    return record


def recheck_metrics(dump):
    """Проверяем агрегацию готовых per-query метрик без повторного подбора моделей."""
    results = []
    for name, table_name in (
        ("lexical_heuristic_dev", "lexical_heuristic_dev.queries.parquet"),
        ("lexical_logloss_dev", "lexical_logloss_dev.queries.parquet"),
        ("lexical_models_dev", "lexical_models_dev.per_query.parquet"),
        ("hybrid_heuristic_dev", "hybrid_heuristic_dev.queries.parquet"),
        ("hybrid_models_dev", "hybrid_models_dev.per_query.parquet"),
        ("history_ablation_dev", "history_ablation_dev.per_query.parquet"),
        ("final_audit", "final_audit.per_query.parquet"),
    ):
        report = json.loads((dump / "reports" / f"{name}.json").read_text())
        rows = pq.read_table(
            dump / "reports" / table_name, use_threads=False
        ).to_pylist()
        models = report.get("results", {"single": report})
        for model_name, expected in models.items():
            selected = (
                rows
                if model_name == "single"
                else [row for row in rows if row["model_name"] == model_name]
            )
            if not selected or len({row["query_id"] for row in selected}) != len(
                selected
            ):
                raise ValueError(f"Некорректное покрытие запросов: {name}/{model_name}")
            for metric in ("recall@10", "recall@50", "recall@100", "recall@union"):
                actual = math.fsum(row[metric] for row in selected) / len(selected)
                difference = abs(actual - expected[metric])
                results.append(
                    {
                        "report": f"data_dump/reports/{name}.json",
                        "table": f"data_dump/reports/{table_name}",
                        "model_name": model_name,
                        "column": metric,
                        "method": "macro mean across unique query_id",
                        "rows": len(selected),
                        "reported": expected[metric],
                        "recomputed": actual,
                        "absolute_difference": difference,
                        "passed": difference <= 1e-12,
                    }
                )
                weighted = "poststratified_" + metric
                if weighted in expected:
                    fraction = expected["reference_empty_fraction"]
                    empty = [row[metric] for row in selected if row["filter_empty"]]
                    filtered = [
                        row[metric] for row in selected if not row["filter_empty"]
                    ]
                    actual = fraction * math.fsum(empty) / len(empty) + (
                        1 - fraction
                    ) * math.fsum(filtered) / len(filtered)
                    difference = abs(actual - expected[weighted])
                    results.append(
                        {
                            "report": f"data_dump/reports/{name}.json",
                            "table": f"data_dump/reports/{table_name}",
                            "model_name": model_name,
                            "column": metric,
                            "method": "p_empty * mean(empty) + (1-p_empty) * mean(filtered)",
                            "reference_empty_fraction": fraction,
                            "rows": len(selected),
                            "reported": expected[weighted],
                            "recomputed": actual,
                            "absolute_difference": difference,
                            "passed": difference <= 1e-12,
                        }
                    )
    return {
        "scope": "Aggregation of saved per-query metrics only. Candidate relevance and the hidden benchmark are not re-evaluated.",
        "absolute_tolerance": 1e-12,
        "all_checks_passed": all(row["passed"] for row in results),
        "checks": results,
    }


def refresh_index(source):
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT):
        raise ValueError("Папка исходного решения должна быть вне папки выдачи")
    dump = ROOT / "data_dump"
    copied = {}
    inputs = []
    for folder in ("reports", "configs"):
        inputs.extend(
            (path, Path(folder) / path.relative_to(source / folder))
            for path in (source / folder).rglob("*")
            if path.is_file() and path.suffix in {".json", ".md", ".parquet", ".log"}
        )
    inputs.extend(
        [
            (source / "docs/task.md", Path("docs/task.md")),
            (source / "deliverables/verification.json", Path("verification.json")),
        ]
    )
    inputs.extend(
        (path, Path("reports") / path.name)
        for path in source.glob("answer*.report.json")
    )
    snapshot = []
    for folder in ("src", "scripts", "tests", "docs"):
        snapshot.extend(
            path
            for path in (source / folder).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".md", ".toml"}
            and not any(part.startswith(".") for part in path.relative_to(source).parts)
        )
    snapshot.extend(source / name for name in ("README.md", "pyproject.toml"))
    snapshot.extend(source.glob("requirements*.txt"))
    snapshot.extend(
        source / relative
        for relative in (
            "reports/final_audit.json",
            "configs/selection.json",
            "reports/lexical_error_analysis.md",
        )
    )
    inputs.extend(
        (path, Path("source_snapshot") / path.relative_to(source)) for path in snapshot
    )
    for path in (source / "artifacts").rglob("*"):
        relative = path.relative_to(source)
        if (
            not path.is_file()
            or "models" in relative.parts
            or any(part.startswith(".") for part in relative.parts)
        ):
            continue
        if (
            path.suffix in {".json", ".log"} and not path.name.endswith(".ids.json")
        ) or path.name.endswith(".dev_queries.parquet"):
            inputs.append((path, Path("experiments") / relative))
    secret = re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|(?i:authorization\s*:\s*(?:bearer|basic)\s+\S+))"
    )
    for original, relative in sorted(inputs, key=lambda pair: pair[1].as_posix()):
        original = safe_path(source, original.relative_to(source))
        if original.suffix == ".log" and secret.search(
            original.read_text(encoding="utf-8", errors="replace")
        ):
            raise ValueError(f"В журнале обнаружен возможный секрет: {original.name}")
        destination = safe_path(dump, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        copied[destination.relative_to(ROOT).as_posix()] = file_record(
            destination, original.relative_to(source).as_posix()
        )

    for name in (
        "independent_audit.json",
        "independent_audit.md",
        "notebook_verification.json",
    ):
        original = safe_path(ROOT, Path("report") / name)
        destination = safe_path(dump, Path("independent") / name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        if name.endswith(".md"):
            destination.write_text(
                original.read_text(encoding="utf-8").replace(
                    "(../evidence/INDEX.md#", "(../../evidence/INDEX.md#"
                ),
                encoding="utf-8",
            )
        record = file_record(destination, f"avito submission/report/{name}")
        if sha256(original) != record["sha256"]:
            record["original_sha256"] = sha256(original)
            record["copy_transform"] = (
                "Rebased evidence links for the relocated Markdown copy"
            )
        copied[destination.relative_to(ROOT).as_posix()] = record

    print("Копии отчётов, конфигураций и исходного кода подготовлены", flush=True)
    with zipfile.ZipFile(source / "deliverables/avito_solution.zip") as archive:
        source_manifest = json.loads(archive.read("manifest.json"))
    artifact_paths = []
    artifact_paths.extend(
        path
        for path in (source / "data").rglob("*")
        if path.is_file() and path.suffix in {".parquet", ".zip"}
    )
    artifact_paths.extend(
        path
        for path in (source / "artifacts").rglob("*")
        if path.is_file()
        and path.suffix in {".npy", ".parquet", ".joblib", ".cbm", ".safetensors"}
        and not any(part.startswith(".") for part in path.relative_to(source).parts)
    )
    artifact_paths.extend(source.glob("answer*.csv"))
    artifact_paths.append(source / "deliverables/avito_solution.zip")
    artifacts = {}
    for original in sorted(set(artifact_paths)):
        original = safe_path(source, original.relative_to(source))
        relative = original.relative_to(source).as_posix()
        record = {
            "sha256": sha256(original),
            "bytes": original.stat().st_size,
            "included_in_evidence_zip": False,
            "member_of_solution_bundle": relative in source_manifest["files"],
        }
        if original.suffix == ".parquet":
            parquet = pq.ParquetFile(original)
            record.update(
                rows=parquet.metadata.num_rows, columns=parquet.schema_arrow.names
            )
        artifacts[relative] = record
    print(f"Хеши тяжёлых артефактов проверены: {len(artifacts)} файлов", flush=True)

    download_code = ast.parse((source / "scripts/download_data.py").read_text())
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in download_code.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"PUBLIC_KEY", "ARCHIVE_SHA256"}
    }
    archive_info = artifacts["data/dataset.zip"]
    if archive_info["sha256"] != constants["ARCHIVE_SHA256"]:
        raise ValueError("Исходный dataset.zip не совпал с закреплённым SHA256")
    dataset = {
        "public_url": constants["PUBLIC_KEY"],
        "archive": {
            "path": "data/dataset.zip",
            "sha256": archive_info["sha256"],
            "expected_sha256": constants["ARCHIVE_SHA256"],
            "bytes": archive_info["bytes"],
            "matches_expected": True,
        },
        "files": {
            name: artifacts[f"data/raw/{name}"]
            for name in (
                "train.parquet",
                "benchmark_queries.parquet",
                "benchmark_items.parquet",
            )
        },
    }
    provenance = json.loads(
        (source / "artifacts/models/multilingual-e5-small/provenance.json").read_text()
    )
    for name, expected in provenance["sha256"].items():
        if (
            sha256(
                safe_path(source, Path("artifacts/models/multilingual-e5-small") / name)
            )
            != expected
        ):
            raise ValueError(f"Хеш файла модели не совпал: {name}")
    verification = json.loads((dump / "verification.json").read_text())
    archive_hash_matches = (
        artifacts["deliverables/avito_solution.zip"]["sha256"]
        == verification["archive_sha256"]
    )
    if not archive_hash_matches:
        raise ValueError(
            "Отчёт воспроизведения относится к другой версии исходного solution-архива"
        )
    independent = json.loads((dump / "independent/independent_audit.json").read_text())
    for relative, record in independent["sources"].items():
        if sha256(safe_path(source, relative)) != record["sha256"]:
            raise ValueError(f"Независимый audit проверял другую версию: {relative}")
    notebook_check = json.loads(
        (dump / "independent/notebook_verification.json").read_text()
    )
    notebook = safe_path(
        ROOT, Path(notebook_check["source_notebook"]).relative_to(ROOT)
    )
    if sha256(notebook) != notebook_check["source_notebook_sha256"]:
        raise ValueError("Мини-ноутбук изменён после независимого запуска")
    for relative, record in notebook_check["copied_inputs"].items():
        if sha256(safe_path(notebook.parent, relative)) != record["sha256"]:
            raise ValueError(f"Вход мини-ноутбука изменён после проверки: {relative}")
    metric_recheck = recheck_metrics(dump)
    if not metric_recheck["all_checks_passed"]:
        raise ValueError("Независимый пересчёт метрик выявил расхождения")

    generated = {
        "dataset.json": dataset,
        "model.json": provenance,
        "artifacts.json": {
            "scope": "Heavy originals remain in the source project or the separate solution bundle. Only checksums are included here.",
            "solution_archive_matches_verification": archive_hash_matches,
            "files": artifacts,
        },
        "source_snapshot.json": {
            "scope": "Read-only copy of the original solution source at index refresh time",
            "files": {
                name: record
                for name, record in copied.items()
                if name.startswith("data_dump/source_snapshot/")
            },
        },
        "training_runs.json": {
            "files": {
                name: record
                for name, record in copied.items()
                if name.endswith(
                    (
                        ".train_config.json",
                        ".evaluation.json",
                        ".feature_names.json",
                        ".dev_queries.parquet",
                    )
                )
            }
        },
        "logs.json": {
            "secret_scan": "Common credential patterns checked before copying",
            "files": {
                name: record for name, record in copied.items() if name.endswith(".log")
            },
        },
        "metric_recheck.json": metric_recheck,
        "solution_manifest.json": source_manifest,
    }
    for name, value in generated.items():
        path = dump / "provenance" / name
        write_json(path, value)
        copied[path.relative_to(ROOT).as_posix()] = file_record(path)
    inventory_path = dump / "provenance/dump_inventory.json"
    write_json(
        inventory_path,
        {"excluded_self": "data_dump/provenance/dump_inventory.json", "files": copied},
    )
    copied[inventory_path.relative_to(ROOT).as_posix()] = file_record(inventory_path)

    entries = []
    loaded = {}
    for identifier, title, relative, kind in CATALOG:
        path = dump / relative
        if not path.is_file():
            raise FileNotFoundError(f"Для {identifier} отсутствует {path}")
        entry = {"id": identifier, "title": title, "kind": kind, **file_record(path)}
        if identifier == "E001":
            document = ROOT.parent.parent / "авито.docx"
            if document.is_file():
                entry["original_document"] = {
                    "path": str(document),
                    "sha256": sha256(document),
                    "bytes": document.stat().st_size,
                    "copy_in_dump": False,
                }
        if identifier == "E002":
            entry["external_url"] = dataset["public_url"]
        if identifier == "E024":
            entry["external_url"] = provenance["model_card"]
        entries.append(entry)
        if path.suffix == ".json":
            loaded[identifier] = json.loads(path.read_text(encoding="utf-8"))
    sources = {
        "schema_version": 1,
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": "avito contest",
        "copy_policy": "Original inputs were read only. Evidence links point to bundled copies.",
        "sources": entries,
    }
    claims = []
    for source_id, pointer, statement, unit, kind in CLAIMS:
        value = json_pointer(loaded[source_id], pointer)
        claims.append(
            {
                "id": f"C{len(claims) + 1:03d}",
                "statement": statement,
                "value": value,
                "unit": unit,
                "kind": kind,
                "source_id": source_id,
                "locator": {"type": "json_pointer", "pointer": pointer},
                "verification_method": "Read the exact JSON pointer in the checksum-verified source copy",
            }
        )
    split_code = (dump / "source_snapshot/scripts/prepare_validation.py").read_text()
    if not any(
        isinstance(node, ast.Constant) and node.value == 0.904
        for node in ast.walk(ast.parse(split_code))
    ):
        raise ValueError("Не найдено допущение о доле скрываемой истории 0.904")
    claims.append(
        {
            "id": f"C{len(claims) + 1:03d}",
            "statement": "Доля 90,4% целей со скрываемой историей задаётся как допущение стресс-проверки, а не измерение скрытого benchmark",
            "value": 0.904,
            "unit": "доля выбранных целевых объявлений",
            "kind": "assumption",
            "source_id": "E027",
            "locator": {
                "type": "source_code",
                "path": "data_dump/source_snapshot/scripts/prepare_validation.py",
                "expression": "round(len(target_ids) * 0.904)",
            },
            "verification_method": "AST contains constant 0.904 in the saved split implementation",
        }
    )
    claims.append(
        {
            "id": f"C{len(claims) + 1:03d}",
            "statement": "Равенство CSV после полного пересчёта E5 на CPU и MPS не установлено",
            "value": None,
            "unit": "не проверено",
            "kind": "limitation",
            "source_id": "E018",
            "locator": {"type": "json_pointer", "pointer": "/cpu_mps_equivalence"},
            "verification_method": "The verification report explicitly limits the result to saved MPS arrays",
        }
    )
    evidence = ROOT / "evidence"
    write_json(evidence / "sources.json", sources)
    write_json(
        evidence / "claims.json",
        {
            "schema_version": 1,
            "kinds": {
                "measurement": "Измерение или рассчитанная метрика",
                "verification": "Проверка в указанном объёме",
                "assumption": "Принятое допущение",
                "design_choice": "Выбранный параметр метода",
                "reported_statement": "Запись в исходном отчёте, не независимое наблюдение",
                "limitation": "Явно непроверенное свойство",
            },
            "claims": claims,
        },
    )
    rows = claim_csv_rows(claims)
    with (evidence / "claims.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Индекс доказательств",
        "",
        "Источники имеют постоянные номера E001, E002 и далее. Ссылки ведут на копии внутри data dump. Полные контрольные суммы и машинные указатели находятся в [sources.json](sources.json) и [claims.json](claims.json). Табличная версия тех же утверждений доступна в [claims.csv](claims.csv). В столбцах value и locator сохранён JSON, чтобы не потерять тип и точность значений.",
        "",
        "Локальный audit не является скрытым результатом платформы. Проверка E020 относится к исходному solution-архиву и трём CSV из него. Новый мини-ноутбук отдельно проверен в E033. Пересчёт E029 проверяет агрегацию сохранённых метрик по запросам и не создаёт новую разметку релевантности.",
        "",
        "## Карта проверки",
        "",
        "1. Сверьте SHA256 исходного набора по E002 и параметры разделения по E006 и E007.",
        "2. Сравните dev-эксперименты E008–E013 с зафиксированным выбором E014.",
        "3. Проверьте audit по E015, итоговые интервалы по E016 и средние по таблицам E029.",
        "4. Проверьте численную повторяемость по E018 и E019, затем совпадение CSV исходного архива по E020 и мини-ноутбука по E033.",
        "5. Сверьте тяжёлые локальные файлы по E026 и исходный код по E027. Архив evidence содержит их контрольные суммы, а не веса или таблицы признаков.",
        "",
        "E003 содержит 73 873 текста после lower, замены ё и свёртки пробелов. E032 содержит 73 706 текстов после полного нормализатора NFKC и regex. Разные определения нормализации дают разные числа и не являются расхождением исходных данных.",
        "",
        "В E032 знаменатель Recall учитывает все сохранённые положительные пары, включая отсутствующие среди кандидатов. Между fit+dev и audit остаются 22 общих целевых item_id. После исключения 22 запросов отдельная проверка чувствительности оставляет 1478 запросов. Общий корпус кандидатов при такой проверке сохраняется.",
        "",
        "## Указатель источников",
        "",
    ]
    for entry in entries:
        identifier = entry["id"]
        lines.extend(
            [
                f'<a id="{identifier}"></a>',
                f'<a id="{identifier.lower()}"></a>',
                f'### {identifier}: {entry["title"]}',
                "",
                f'[Открыть источник](../{entry["path"]})',
                "",
                f'Тип: `{entry["kind"]}`. Размер: {entry["bytes"]} байт.',
                "",
                f'SHA256: `{entry["sha256"]}`',
                "",
            ]
        )
        if "external_url" in entry:
            lines.extend([f'[Внешний первоисточник]({entry["external_url"]})', ""])
        if "original_document" in entry:
            lines.extend(
                [
                    "Хеш предоставленного документа авито.docx: "
                    f'`{entry["original_document"]["sha256"]}`.',
                    "",
                ]
            )
    lines.extend(
        [
            "## Проверяемые утверждения",
            "",
            "Значения ниже сохраняют точность исходных JSON. JSON Pointer использует правила RFC 6901. Строка с типом assumption является допущением, reported_statement отражает запись автора исходного отчёта.",
            "",
            "| ID | Утверждение | Значение | Единица | Тип | Источник и указатель |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for claim in claims:
        value = json.dumps(claim["value"], ensure_ascii=False)
        locator = claim["locator"].get(
            "pointer", claim["locator"].get("expression", "")
        )
        lines.append(
            f'| {claim["id"]} | {claim["statement"]} | `{value}` | {claim["unit"]} | {claim["kind"]} | [{claim["source_id"]}](#{claim["source_id"].lower()}) `{locator}` |'
        )
    lines.extend(
        [
            "",
            "## Состав data dump",
            "",
            "Все исходные отчёты JSON, Markdown и Parquet сохранены в data_dump/reports. Параметры обучения, кривые и журналы находятся в data_dump/experiments. Снимок кода находится в data_dump/source_snapshot. E030 перечисляет каждый включённый файл с размером и SHA256.",
            "",
            "Полные исходные тексты, dataset.zip, веса, эмбеддинги и таблицы кандидатов не копируются в evidence-архив. Их местоположение и контрольные суммы перечислены в E026. Отдельный solution bundle содержит сохранённые benchmark-признаки и модели для точного повторения CSV.",
            "",
            "Самохеширование исключено. E030 не включает собственный хеш, sources.json не ссылается на свой хеш, а архивная опись не включает собственный файл. Финальный SHA256 ZIP выводится отдельно после записи.",
            "",
        ]
    )
    index_text = "\n".join(lines)
    if chr(0x2014) in index_text or chr(59) in index_text:
        raise ValueError("В новой прозе обнаружен запрещённый знак")
    (evidence / "INDEX.md").write_text(index_text, encoding="utf-8")
    payload = [ROOT / name for name in copied]
    payload.extend(
        [
            evidence / "INDEX.md",
            evidence / "sources.json",
            evidence / "claims.json",
            evidence / "claims.csv",
            Path(__file__).resolve(),
        ]
    )
    return payload, {
        "sources": len(entries),
        "claims": len(claims),
        "dump_files": len(copied),
        "metric_checks": len(metric_recheck["checks"]),
        "all_metric_checks_passed": metric_recheck["all_checks_passed"],
    }


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("evidence_manifest.json"))
        expected = set(manifest["files"]) | {"evidence_manifest.json"}
        if (
            len(archive.namelist()) != len(expected)
            or set(archive.namelist()) != expected
        ):
            raise ValueError("Набор файлов evidence-архива не совпадает с описью")
        for name, record in manifest["files"].items():
            safe_path(ROOT, name)
            info = archive.getinfo(name)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Символическая ссылка в архиве: {name}")
            with archive.open(name) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != record["sha256"] or info.file_size != record["bytes"]:
                raise ValueError(f"Не совпал файл evidence-архива: {name}")
        claims = json.loads(archive.read("evidence/claims.json"))["claims"]
        csv_text = archive.read("evidence/claims.csv").decode("utf-8")
        csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
        if csv_rows != claim_csv_rows(claims):
            raise ValueError("CSV и JSON реестры утверждений не совпали")
    return {
        "verified": True,
        "files": len(manifest["files"]),
        "sha256": sha256(path),
        "bytes": Path(path).stat().st_size,
        "claims_csv_rows": len(csv_rows),
        "claims_csv_matches_json": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Обновить копии и индекс без архива, как при запуске без аргументов",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="После обновления индекса явно собрать ZIP",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "deliverables/avito_evidence.zip"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Проверить готовый ZIP без доступа к исходному решению",
    )
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_archive(args.verify), ensure_ascii=False, indent=2))
        return
    payload, summary = refresh_index(args.source)
    if args.package:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT) or output.suffix != ".zip":
            raise ValueError(
                "Evidence-архив должен находиться внутри папки выдачи и иметь расширение .zip"
            )
        output = safe_path(ROOT, output.relative_to(ROOT))
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": 1,
            "purpose": "evidence_data_dump",
            "self_hash_excluded": "evidence_manifest.json",
            "files": {
                path.relative_to(ROOT).as_posix(): {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(payload)
            },
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, suffix=".zip", delete=False
            ) as handle:
                temporary = Path(handle.name)
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for path in sorted(payload):
                    archive.write(path, path.relative_to(ROOT).as_posix())
                archive.writestr(
                    "evidence_manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                )
            summary["archive"] = verify_archive(temporary)
            os.replace(temporary, output)
            summary["archive"]["path"] = str(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        summary["archive_created"] = False
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
