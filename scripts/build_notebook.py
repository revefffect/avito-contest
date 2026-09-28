"""Собрать короткий ноутбук применения готовых моделей."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown(r"""
# Поиск объявлений Авито

Ноутбук заново получает основной `answer.csv` из сохранённых признаков и двух
обученных моделей. Готовый ответ не читается. Контрольная сумма в конце проверяет
точное совпадение с выпущенным результатом.

Для запуска нужны файлы из этой папки и библиотеки из `requirements.txt`.
После установки зависимостей выберите **Restart Kernel and Run All Cells**.
Сеть и исходные тексты объявлений при этом не используются.

Полная подготовка данных и обучение находятся в
[исходном решении](../reference_solution/README.md).
[Отчёт](../report/method.md) объясняет поиск и ограничения оценки.
[Индекс источников](../evidence/INDEX.md) связывает числа с файлами экспериментов.

Здесь повторяется применение уже обученных моделей. Повторное обучение E5 или
CatBoost в этот короткий запуск не входит.
""")

markdown(r"""
## 1. Проверить комплект

В таблице кандидатов уже рассчитаны текстовые, географические и поведенческие
признаки. Модели не получают `query_id`, `item_id`, метку ответа или номер строки
объявления. Позиционные индексы нужны только для связи таблиц.

Проверим контрольные суммы до расчёта. Так случайная подмена файла или другой
порядок объявлений не останутся незаметными.
""")

code(r"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from catboost import CatBoost

ROOT = Path.cwd()
if not (ROOT / "config.json").exists() and (ROOT / "mini/config.json").exists():
    ROOT = ROOT / "mini"
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for name, expected in config["files"].items():
    actual = sha256(ROOT / name)
    assert actual == expected, f"Изменился входной файл: {name}"

print(f"Проверено файлов: {len(config['files'])}")
""")

markdown(r"""
## 2. Загрузить кандидатов

Один запрос имеет несколько сотен кандидатов. После оценки останутся ровно 50.
Исходные идентификаторы сохраняются строками, включая ведущие нули и регистр.
""")

code(r"""
queries = pd.read_parquet(ROOT / "data/queries.parquet", use_threads=False)
items = pd.read_parquet(ROOT / "data/items.parquet", use_threads=False)
features = pd.read_parquet(ROOT / "data/features.parquet", use_threads=False)

assert queries["query_id"].is_unique
assert queries["query_id"].map(lambda value: isinstance(value, str) and len(value) == 16).all()
assert items["item_id"].is_unique
assert items["item_id"].map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{16}", value))).all()
assert not features[["qid", "item_idx"]].duplicated().any()
assert pd.api.types.is_integer_dtype(features["qid"])
assert pd.api.types.is_integer_dtype(features["item_idx"])
assert set(features["qid"]) == set(range(len(queries)))
assert features["item_idx"].between(0, len(items) - 1).all()

qids = features["qid"].to_numpy()
item_indices = features["item_idx"].to_numpy()
counts = np.bincount(qids, minlength=len(queries))
assert (counts >= 50).all()
pd.Series({"Запросы": len(queries), "Объявления корпуса": len(items), "Пары для оценки": len(features)})
""")

markdown(r"""
## 3. Применить две модели

Первая модель использует прямые счётчики выбора объявлений. Во второй эти
признаки исключены. Обе модели оценивают один набор кандидатов.

Масштабы оценок могут различаться. Для каждого запроса отдельно вычтем среднее
и разделим результат на стандартное отклонение. Если все оценки одинаковы,
нормированная оценка равна нулю.

$$
z_m(q,d)=\frac{s_m(q,d)-\mu_m(q)}{\sigma_m(q)},
\qquad s(q,d)=0.5z_1(q,d)+0.5z_2(q,d).
$$

Среднее и отклонение считаются по всем кандидатам запроса. Параметры смеси
зафиксированы до итоговой независимой проверки [E014](../evidence/INDEX.md).
""")

code(r"""
def standardize(scores):
    means = np.bincount(qids, weights=scores, minlength=len(queries)) / counts
    centered = scores - means[qids]
    variance = np.bincount(qids, weights=centered * centered, minlength=len(queries)) / counts
    scale = np.sqrt(variance[qids])
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)


combined = np.zeros(len(features), dtype=np.float64)
reserved = {"qid", "item_idx", "query_id", "item_id", "label", "split", "target"}
weights = np.array([record["weight"] for record in config["models"]], dtype=float)
assert np.isfinite(weights).all() and (weights >= 0).all()
assert np.isfinite(weights.sum()) and weights.sum() > 0
weights /= weights.sum()

for record, weight in zip(config["models"], weights):
    names = json.loads((ROOT / record["features"]).read_text())
    assert len(names) == len(set(names)) and not reserved.intersection(names)
    assert np.isfinite(features[names].to_numpy()).all()
    model = CatBoost()
    model.load_model(ROOT / record["path"])
    assert names == model.feature_names_
    scores = np.asarray(model.predict(features[names], prediction_type="RawFormulaVal", thread_count=4), dtype=np.float64)
    assert np.isfinite(scores).all()
    combined += weight * standardize(scores)
    print(f"{Path(record['path']).name}: {model.tree_count_} деревьев, {len(names)} признаков")
""")

markdown(r"""
## 4. Выбрать 50 объявлений

При равных оценках используем исходную позицию объявления. Такой порядок
делает результат повторяемым. Для Recall@50 важен состав списка, а не порядок
объявлений внутри него.
""")

code(r"""
order = np.lexsort((item_indices, -combined, qids))
sorted_qids = qids[order]
starts = np.r_[0, np.flatnonzero(np.diff(sorted_qids)) + 1]
ends = np.r_[starts[1:], len(order)]
item_ids = items["item_id"].to_numpy()
answers = [None] * len(queries)

for start, end in zip(starts, ends):
    selected = item_indices[order[start:min(start + 50, end)]]
    answers[int(sorted_qids[start])] = " ".join(item_ids[selected])

answer = pd.DataFrame({"query_id": queries["query_id"], "answer": answers})
assert answer["answer"].notna().all()
allowed = set(item_ids)
for value in answer["answer"]:
    selected = value.split()
    assert len(selected) == len(set(selected)) == 50
    assert set(selected).issubset(allowed)
""")

markdown(r"""
## 5. Сохранить и проверить ответ

CSV содержит только `query_id` и `answer`. После записи снова прочитаем файл
как строки и сравним SHA256 с основным ответом. Контрольная сумма проверяет
весь файл, включая порядок строк и формат записи.
""")

code(r"""
output = ROOT / "answer.csv"
temporary = ROOT / "answer.csv.part"
answer.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
restored = pd.read_csv(temporary, dtype=str, keep_default_na=False)
assert restored.equals(answer)
actual = sha256(temporary)
assert actual == config["answer_sha256"], "Ответ отличается от зафиксированного результата"
temporary.replace(output)
print(f"Готово: {len(answer)} запросов, по 50 объявлений")
print(f"SHA256: {actual}")
""")

markdown(r"""
## Как оценивалось качество

Разметки benchmark нет, поэтому этот ноутбук не вычисляет его Recall.
На 1 500 независимых локальных запросах основной ансамбль получил
Recall@50 = 0.941111. Число взято из выполненного эксперимента
[E015](../evidence/INDEX.md), а не рассчитано по текущим неразмеченным запросам.

$$
\operatorname{Recall@50}=\frac{1}{|Q|}\sum_{q\in Q}
\frac{|T_{50}(q)\cap R(q)|}{|R(q)|}.
$$

В знаменателе стоит вся известная разметка запроса, включая объявления,
которые не попали в кандидаты. На скрытом тесте результат может отличаться.
Подробности разбиения, анализ ошибок и независимые сверки доступны
в [отчёте](../report/method.md) и [индексе доказательств](../evidence/INDEX.md).
""")

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.7"},
    },
)
nbf.write(notebook, ROOT / "mini/solution.ipynb")
