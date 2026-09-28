# Индекс доказательств

Источники имеют постоянные номера E001, E002 и далее. Ссылки ведут на копии внутри data dump. Полные контрольные суммы и машинные указатели находятся в [sources.json](sources.json) и [claims.json](claims.json). Табличная версия тех же утверждений доступна в [claims.csv](claims.csv). В столбцах value и locator сохранён JSON, чтобы не потерять тип и точность значений.

Локальный audit не является скрытым результатом платформы. Проверка E020 относится к исходному solution-архиву и трём CSV из него. Новый мини-ноутбук отдельно проверен в E033. Пересчёт E029 проверяет агрегацию сохранённых метрик по запросам и не создаёт новую разметку релевантности.

## Карта проверки

1. Сверьте SHA256 исходного набора по E002 и параметры разделения по E006 и E007.
2. Сравните dev-эксперименты E008–E013 с зафиксированным выбором E014.
3. Проверьте audit по E015, итоговые интервалы по E016 и средние по таблицам E029.
4. Проверьте численную повторяемость по E018 и E019, затем совпадение CSV исходного архива по E020 и мини-ноутбука по E033.
5. Сверьте тяжёлые локальные файлы по E026 и исходный код по E027. Архив evidence содержит их контрольные суммы, а не веса или таблицы признаков.

E003 содержит 73 873 текста после lower, замены ё и свёртки пробелов. E032 содержит 73 706 текстов после полного нормализатора NFKC и regex. Разные определения нормализации дают разные числа и не являются расхождением исходных данных.

В E032 знаменатель Recall учитывает все сохранённые положительные пары, включая отсутствующие среди кандидатов. Между fit+dev и audit остаются 22 общих целевых item_id. После исключения 22 запросов отдельная проверка чувствительности оставляет 1478 запросов. Общий корпус кандидатов при такой проверке сохраняется.

## Указатель источников

<a id="E001"></a>
<a id="e001"></a>
### E001: Условие конкурса

[Открыть источник](../data_dump/docs/task.md)

Тип: `specification`. Размер: 13835 байт.

SHA256: `70fb3c5965386fac86487f56d87fa5114f8f5aa87f8c609fd5bf0995c3fc28da`

Хеш предоставленного документа авито.docx: `ac0e3a19c0f185629f3b32af85a0cefb6c2c7e91c5186a74795dd93b46060e47`.

<a id="E002"></a>
<a id="e002"></a>
### E002: Происхождение и контрольные суммы набора данных

[Открыть источник](../data_dump/provenance/dataset.json)

Тип: `provenance`. Размер: 2305 байт.

SHA256: `c585de27b090b7e9f953966b620e2c43db01f3cc0d2c6bb022ba6d19342f5ad0`

[Внешний первоисточник](https://disk.yandex.ru/d/sNhfo0YOjGtufg)

<a id="E003"></a>
<a id="e003"></a>
### E003: Профиль исходных данных

[Открыть источник](../data_dump/reports/data_profile.json)

Тип: `measurement`. Размер: 9970 байт.

SHA256: `975bafbceca6393a829d4c210dbbf9f25565bd1113ec52a794b51a6d41cb670e`

<a id="E004"></a>
<a id="e004"></a>
### E004: География и фильтры

[Открыть источник](../data_dump/reports/geography_analysis.json)

Тип: `measurement`. Размер: 39171 байт.

SHA256: `eefcb1f4964e02c5c7fe4bc3c52cd4d0874f26f74c972b257439db2e8e94f73a`

<a id="E005"></a>
<a id="e005"></a>
### E005: Сдвиг категорий поиска

[Открыть источник](../data_dump/reports/category_shift.json)

Тип: `measurement`. Размер: 580 байт.

SHA256: `f35c60257aab620b7f18ed3e9c7fcd7942807656f33a374fa3e494c4fe09031e`

<a id="E006"></a>
<a id="e006"></a>
### E006: Локальное разделение данных

[Открыть источник](../data_dump/reports/validation_split.json)

Тип: `measurement`. Размер: 1143 байт.

SHA256: `5c04e5c0854b3df2dcf70525a2ef395823a2f52b7159f1c2dbfeb3ac74a71730`

<a id="E007"></a>
<a id="e007"></a>
### E007: Повторное построение разделения

[Открыть источник](../data_dump/reports/split_reproducibility.json)

Тип: `verification`. Размер: 2769 байт.

SHA256: `0022343991c220d71f292e9403386614d2ad8a56989c28e5b9e808c53ae239bf`

<a id="E008"></a>
<a id="e008"></a>
### E008: Лексическая эвристика на dev

[Открыть источник](../data_dump/reports/lexical_heuristic_dev.json)

Тип: `measurement`. Размер: 2982 байт.

SHA256: `924aacc1d23f3759c9169f3d91db8dcd3786b097aa63307d422d7ad976160fc2`

<a id="E009"></a>
<a id="e009"></a>
### E009: Лексическая модель Logloss на dev

[Открыть источник](../data_dump/reports/lexical_logloss_dev.json)

Тип: `measurement`. Размер: 3296 байт.

SHA256: `c258cff16ea288407186fe544223fdad14ebabfc86300f1960b14e422a3a1425`

<a id="E010"></a>
<a id="e010"></a>
### E010: Сравнение лексических моделей

[Открыть источник](../data_dump/reports/lexical_models_dev.json)

Тип: `measurement`. Размер: 9644 байт.

SHA256: `d88310c490def05f157abc877d70bb7d6db816e42f251d516745ecf7aec01f95`

<a id="E011"></a>
<a id="e011"></a>
### E011: Гибридная эвристика на dev

[Открыть источник](../data_dump/reports/hybrid_heuristic_dev.json)

Тип: `measurement`. Размер: 3282 байт.

SHA256: `faf7f960ac99ba5e6426f768de54d3f278714d7cae3470e78022f260f212b6e2`

<a id="E012"></a>
<a id="e012"></a>
### E012: Сравнение гибридных моделей

[Открыть источник](../data_dump/reports/hybrid_models_dev.json)

Тип: `measurement`. Размер: 22235 байт.

SHA256: `56f905d29cacf285bf658a0e8fd85ab6c4842a58e6739e21ddb2ff5898f0dcec`

<a id="E013"></a>
<a id="e013"></a>
### E013: Исключение прямой истории выбора

[Открыть источник](../data_dump/reports/history_ablation_dev.json)

Тип: `measurement`. Размер: 22184 байт.

SHA256: `37b954fbe49d49c342712132d98103cff59669f95937e4595293fa36f9caf220`

<a id="E014"></a>
<a id="e014"></a>
### E014: Протокол выбора до audit

[Открыть источник](../data_dump/configs/selection.json)

Тип: `protocol`. Размер: 2125 байт.

SHA256: `0794eeeae97538bd95c2827c4f1feb3ddba75522365a8b308386a8b4ac4a5601`

<a id="E015"></a>
<a id="e015"></a>
### E015: Результат независимой локальной проверки

[Открыть источник](../data_dump/reports/final_audit.json)

Тип: `measurement`. Размер: 13600 байт.

SHA256: `99c52cb9bf0e9395fd611eaa9abf34d7a0b5256e121b74485f47e65aad90da7b`

<a id="E016"></a>
<a id="e016"></a>
### E016: Итоговые результаты и интервалы

[Открыть источник](../data_dump/reports/final_results.json)

Тип: `measurement`. Размер: 5186 байт.

SHA256: `236db753ec7c223ee2e313a633a5d1b3f994451eaec10b565a490abfa1d04629`

<a id="E017"></a>
<a id="e017"></a>
### E017: Окружение вычислений

[Открыть источник](../data_dump/reports/runtime.json)

Тип: `runtime`. Размер: 763 байт.

SHA256: `0d71705ce12d74dbe3d05b774647214ea1b1ce4a45a03ee5884da2945031bc04`

<a id="E018"></a>
<a id="e018"></a>
### E018: Проверка пяти массивов E5

[Открыть источник](../data_dump/reports/embedding_verification.json)

Тип: `verification`. Размер: 6546 байт.

SHA256: `618b9e16102dbf907658171d411128cd440a387be9a9ae79486e870f54193dc0`

<a id="E019"></a>
<a id="e019"></a>
### E019: Повторяемость признаков на трёх запросах

[Открыть источник](../data_dump/reports/feature_reproducibility.json)

Тип: `verification`. Размер: 13575 байт.

SHA256: `24fea7408bf2b7cc1b2636dc8126253b8f3133e8d930c13378b3411f67999015`

<a id="E020"></a>
<a id="e020"></a>
### E020: Повторное получение трёх CSV из исходного архива

[Открыть источник](../data_dump/verification.json)

Тип: `verification`. Размер: 4776 байт.

SHA256: `afaeba9e242aa33f11d3ee6ef567c1f47d94f511a7386cfce00544bdebc89de3`

<a id="E021"></a>
<a id="e021"></a>
### E021: Финальная конфигурация решения

[Открыть источник](../data_dump/configs/final.json)

Тип: `configuration`. Размер: 2415 байт.

SHA256: `23aed16f25a3a087aa9bb1761b655056f85258d193d403cc5e80e7bf1e8575dc`

<a id="E022"></a>
<a id="e022"></a>
### E022: Контрольные суммы таблиц идентификаторов

[Открыть источник](../data_dump/reports/portable_ids.json)

Тип: `verification`. Размер: 518 байт.

SHA256: `89af144dee04d635a2b1c5bf743c6515ef81414967ad3f318ffaf961a1c00de1`

<a id="E023"></a>
<a id="e023"></a>
### E023: Разбор ошибок лексического поиска

[Открыть источник](../data_dump/reports/lexical_error_analysis.md)

Тип: `analysis`. Размер: 12240 байт.

SHA256: `6398733e8e61565feaea70be00c5bce2aa03d16fa23f40dcf225968c197098be`

<a id="E024"></a>
<a id="e024"></a>
### E024: Источник открытой модели E5

[Открыть источник](../data_dump/provenance/model.json)

Тип: `provenance`. Размер: 906 байт.

SHA256: `ffc878df9f64d21a8633e32f39adc0cfd71cfa4b0cbe10eb2eac10cc747f23a9`

[Внешний первоисточник](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md)

<a id="E025"></a>
<a id="e025"></a>
### E025: Параметры обучения и кривые экспериментов

[Открыть источник](../data_dump/provenance/training_runs.json)

Тип: `inventory`. Размер: 10774 байт.

SHA256: `d2e9b844c9a0c34f9ed7b14709eab7669fbfdcefb9f6d4f35777b4381370b1d0`

<a id="E026"></a>
<a id="e026"></a>
### E026: Хеши данных, индексов, векторов, моделей и ответов

[Открыть источник](../data_dump/provenance/artifacts.json)

Тип: `inventory`. Размер: 21143 байт.

SHA256: `a8a67caeb98e3158d325a2da6f9a191ba70dd4f0288ecc3288822e4864d8729a`

<a id="E027"></a>
<a id="e027"></a>
### E027: Снимок исходного кода и его контрольные суммы

[Открыть источник](../data_dump/provenance/source_snapshot.json)

Тип: `inventory`. Размер: 11154 байт.

SHA256: `b20790b88a32a4e4504889c1c8e44e262fb4e79ca0fce0e1626565d398477f9c`

<a id="E028"></a>
<a id="e028"></a>
### E028: Сохранённые журналы экспериментов

[Открыть источник](../data_dump/provenance/logs.json)

Тип: `inventory`. Размер: 7651 байт.

SHA256: `44653fc5f4a3c7646de9aec4022bd9b54aae628a5da969f490a7af233a36d0a9`

<a id="E029"></a>
<a id="e029"></a>
### E029: Независимый пересчёт средних метрик по таблицам запросов

[Открыть источник](../data_dump/provenance/metric_recheck.json)

Тип: `verification`. Размер: 65100 байт.

SHA256: `d223f9c59e8ff5e9cb16bc9e9e43154ac55d586ab8613f59873bb51328381a10`

<a id="E030"></a>
<a id="e030"></a>
### E030: Полная опись файлов data dump

[Открыть источник](../data_dump/provenance/dump_inventory.json)

Тип: `inventory`. Размер: 48070 байт.

SHA256: `4a69e7bf7e8fc40428c22037cced3597c9fdd7d8b66ce45863f16b7fe5aeb574`

<a id="E031"></a>
<a id="e031"></a>
### E031: Опись исходного архива решения

[Открыть источник](../data_dump/provenance/solution_manifest.json)

Тип: `provenance`. Размер: 11172 байт.

SHA256: `2a16748c82596a4b5b1a608644c2ea83dd38a089146c982c7a37ceef353290c3`

<a id="E032"></a>
<a id="e032"></a>
### E032: Независимая проверка метрик и отсутствия утечки

[Открыть источник](../data_dump/independent/independent_audit.json)

Тип: `verification`. Размер: 27953 байт.

SHA256: `28f823d0460da4d76743b6bc53d56f569b04cee894c1283eb7e2789cc99483a8`

<a id="E033"></a>
<a id="e033"></a>
### E033: Фактический запуск отдельного мини-ноутбука

[Открыть источник](../data_dump/independent/notebook_verification.json)

Тип: `verification`. Размер: 4099 байт.

SHA256: `a07ebf211ae189c8f7a52df46b43fa8a6544c5c47bd56e7c71abfa7756580c66`

<a id="E034"></a>
<a id="e034"></a>
### E034: Проверка основного benchmark-ответа

[Открыть источник](../data_dump/reports/answer.report.json)

Тип: `verification`. Размер: 1514 байт.

SHA256: `3f75c3bd73b71c3d3293d138b5991ea4f5940d6e5cc3f0f6f564b9cbfb4cfbee`

## Проверяемые утверждения

Значения ниже сохраняют точность исходных JSON. JSON Pointer использует правила RFC 6901. Строка с типом assumption является допущением, reported_statement отражает запись автора исходного отчёта.

| ID | Утверждение | Значение | Единица | Тип | Источник и указатель |
| --- | --- | --- | --- | --- | --- |
| C001 | SHA256 исходного архива набора данных | `"8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777"` | SHA256 | measurement | [E002](#e002) `/archive/sha256` |
| C002 | Число строк train | `497673` | строк | measurement | [E002](#e002) `/files/train.parquet/rows` |
| C003 | Число benchmark-запросов | `2452` | запросов | measurement | [E002](#e002) `/files/benchmark_queries.parquet/rows` |
| C004 | Число объявлений benchmark | `189212` | объявлений | measurement | [E002](#e002) `/files/benchmark_items.parquet/rows` |
| C005 | Число уникальных объявлений в train | `344825` | объявлений | measurement | [E003](#e003) `/unique_train_items` |
| C006 | Число текстов train после lower, замены ё на е и свёртки пробелов | `73873` | текстов | measurement | [E003](#e003) `/unique_train_texts` |
| C007 | Доля benchmark-запросов со знакомым текстом | `0.3735725938009788` | доля | measurement | [E003](#e003) `/benchmark_seen_text_fraction` |
| C008 | Доля знакомых полных контекстов benchmark | `0.04404567699836868` | доля | measurement | [E003](#e003) `/benchmark_seen_context_fraction` |
| C009 | Доля объявлений benchmark, встречавшихся в train | `0.09588186795763483` | доля корпуса | measurement | [E003](#e003) `/corpus_items_seen_in_train` |
| C010 | Доля train-пар с совпадающей локацией | `0.8310195650557696` | доля пар | measurement | [E003](#e003) `/train_search_item_same_location` |
| C011 | Доля benchmark-запросов с объявлением той же локации в корпусе | `0.825856` | доля запросов | measurement | [E004](#e004) `/benchmark_query_fraction_with_exact_corpus_location` |
| C012 | Число пар с распознанным видом услуги | `320264` | пар | measurement | [E004](#e004) `/parsed_known_service_filter_pairs` |
| C013 | Доля совпадений распознанного вида услуги | `0.983642` | доля пар | measurement | [E004](#e004) `/parsed_service_pair_match_fraction` |
| C014 | Число train-строк с категорией поиска 0 | `34` | строк | measurement | [E005](#e005) `/train_search_category_zero_rows` |
| C015 | Число benchmark-запросов с категорией поиска 0 | `222` | запросов | measurement | [E005](#e005) `/benchmark_search_category_zero_rows` |
| C016 | Seed локального разделения | `20260928` | целое число | design_choice | [E006](#e006) `/seed` |
| C017 | Общее число отложенных контекстов | `6000` | запросов | measurement | [E006](#e006) `/queries` |
| C018 | Число положительных отложенных пар | `6432` | пар | measurement | [E006](#e006) `/truth_pairs` |
| C019 | Число запросов fit | `3000` | запросов | measurement | [E006](#e006) `/split_counts/fit` |
| C020 | Число запросов dev | `1500` | запросов | measurement | [E006](#e006) `/split_counts/dev` |
| C021 | Число запросов audit | `1500` | запросов | measurement | [E006](#e006) `/split_counts/audit` |
| C022 | Размер полного корпуса локальной проверки | `195125` | объявлений | measurement | [E006](#e006) `/validation_corpus_size` |
| C023 | Число добавленных отложенных объявлений | `5913` | объявлений | measurement | [E006](#e006) `/extra_validation_items` |
| C024 | Фактическая доля знакомых целевых объявлений после удаления истории | `0.03435945273631841` | доля целей | measurement | [E006](#e006) `/seen_target_item_fraction` |
| C025 | Повторное построение дало те же четыре Parquet-файла по хешам | `true` | логическое значение | verification | [E007](#e007) `/all_files_match` |
| C026 | Recall@50 лексической эвристики на dev | `0.8588333333333333` | доля | measurement | [E008](#e008) `/recall@50` |
| C027 | Полнота объединения лексических кандидатов на dev | `0.9667777777777776` | доля | measurement | [E008](#e008) `/recall@union` |
| C028 | Recall@50 лексического Logloss на dev | `0.929888888888889` | доля | measurement | [E009](#e009) `/recall@50` |
| C029 | Recall@50 лексического QuerySoftMax на dev | `0.925388888888889` | доля | measurement | [E010](#e010) `/results/ranker_lexical_softmax/recall@50` |
| C030 | Recall@50 гибридной эвристики на dev | `0.8781111111111111` | доля | measurement | [E011](#e011) `/recall@50` |
| C031 | Полнота объединения гибридных кандидатов на dev | `0.9893333333333333` | доля | measurement | [E011](#e011) `/recall@union` |
| C032 | Recall@50 гибридной модели глубины 6 на dev | `0.942611111111111` | доля | measurement | [E012](#e012) `/results/ranker_hybrid_d6/recall@50` |
| C033 | Recall@50 гибридной модели глубины 7 на dev | `0.9406111111111112` | доля | measurement | [E012](#e012) `/results/ranker_hybrid_d7/recall@50` |
| C034 | Recall@50 модели без прямых счётчиков истории на dev | `0.9391666666666667` | доля | measurement | [E013](#e013) `/results/ranker_hybrid_no_history/recall@50` |
| C035 | Recall@50 выбранной равной смеси на dev | `0.9429444444444444` | доля | measurement | [E013](#e013) `/results/blend_0.5_0.5/recall@50` |
| C036 | Recall@50 выбранной смеси с поправкой на долю пустых фильтров на dev | `0.9371367695172583` | доля | measurement | [E013](#e013) `/results/blend_0.5_0.5/poststratified_recall@50` |
| C037 | Время фиксации решения, записанное в протоколе выбора | `"2026-09-28T03:01:24.152451+00:00"` | UTC | reported_statement | [E014](#e014) `/frozen_at_utc` |
| C038 | Флаг просмотра audit, записанный при выборе | `false` | логическое значение | reported_statement | [E014](#e014) `/audit_quality_seen` |
| C039 | Разность dev Recall@50 между смесью и моделью глубины 6 | `0.0003333333333333333` | доля | measurement | [E014](#e014) `/delta_vs_d6/recall@50/delta` |
| C040 | Парный bootstrap-интервал разности dev Recall@50 | `[-0.004333333333333333, 0.005]` | 95% интервал в долях | measurement | [E014](#e014) `/delta_vs_d6/recall@50/ci95` |
| C041 | Число запросов финального обучения fit и dev | `4500` | запросов | design_choice | [E014](#e014) `/final_training_queries` |
| C042 | Recall@50 выбранной смеси на независимом локальном audit | `0.941111111111111` | доля | measurement | [E015](#e015) `/results/blend_0.5_0.5/recall@50` |
| C043 | Recall@50 смеси на audit с поправкой на пустые фильтры | `0.9431778112629854` | доля | measurement | [E015](#e015) `/results/blend_0.5_0.5/poststratified_recall@50` |
| C044 | Полнота объединённого набора на audit | `0.9886666666666667` | доля | measurement | [E015](#e015) `/results/blend_0.5_0.5/recall@union` |
| C045 | Recall@50 финальной модели с историей на audit | `0.9397777777777777` | доля | measurement | [E015](#e015) `/results/final_d6/recall@50` |
| C046 | Recall@50 финальной модели без прямых счётчиков истории на audit | `0.9464166666666667` | доля | measurement | [E015](#e015) `/results/final_no_history/recall@50` |
| C047 | Bootstrap-интервал audit Recall@50 выбранной смеси | `[0.9291083333333332, 0.9524444444444443]` | 95% интервал в долях | measurement | [E016](#e016) `/audit_recall50_ci95/blend_0.5_0.5` |
| C048 | Число bootstrap-выборок для итоговых интервалов | `5000` | выборок | design_choice | [E016](#e016) `/bootstrap_samples` |
| C049 | Флаг изменения выбора после audit в итоговом отчёте | `false` | логическое значение | reported_statement | [E016](#e016) `/selection_changed_after_audit` |
| C050 | Скрытый результат платформы не измерен | `null` | неизвестно | reported_statement | [E016](#e016) `/hidden_benchmark_score` |
| C051 | Число использованных попыток по итоговому отчёту | `0` | попыток | reported_statement | [E016](#e016) `/submission_attempts_used` |
| C052 | Число строк основного ответа | `2452` | строк | measurement | [E016](#e016) `/answers/answer.csv/rows` |
| C053 | Минимальное число кандидатов в строке основного ответа | `50` | объявлений | measurement | [E016](#e016) `/answers/answer.csv/min_items` |
| C054 | Максимальное число кандидатов в строке основного ответа | `50` | объявлений | measurement | [E016](#e016) `/answers/answer.csv/max_items` |
| C055 | SHA256 основного CSV | `"3ebda47a966aa2920fe9480cb3b4b1602ec91b63f207d89775d981d16995e3c6"` | SHA256 | verification | [E016](#e016) `/answers/answer.csv/sha256` |
| C056 | Версия Python выполненных экспериментов | `"3.12.7"` | версия | measurement | [E017](#e017) `/python` |
| C057 | Устройство расчёта сохранённых эмбеддингов | `"mps"` | устройство | measurement | [E017](#e017) `/embedding_device` |
| C058 | Численная точность сохранённого расчёта E5 | `"float32"` | тип | measurement | [E017](#e017) `/embedding_precision` |
| C059 | Число тестов, указанное в сохранённом runtime-отчёте | `67` | тестов | reported_statement | [E017](#e017) `/tests/passed` |
| C060 | Результат проверки пяти массивов E5 | `"passed"` | статус | verification | [E018](#e018) `/status` |
| C061 | Закреплённая ревизия модели E5 | `"614241f622f53c4eeff9890bdc4f31cfecc418b3"` | git revision | verification | [E018](#e018) `/model_revision` |
| C062 | Форма матрицы эмбеддингов корпуса | `[195125, 384]` | строки и размерность | measurement | [E018](#e018) `/outputs/0/shape` |
| C063 | Максимальное отклонение нормы векторов корпуса от единицы | `1.7881393432617188e-07` | абсолютная ошибка | verification | [E018](#e018) `/outputs/0/max_norm_error` |
| C064 | Число запросов проверки повторяемости признаков | `3` | запросов | verification | [E019](#e019) `/compared_query_count` |
| C065 | Число строк признаков в проверке повторяемости | `3400` | строк | verification | [E019](#e019) `/saved_rows` |
| C066 | Максимальная численная разность повторно построенных признаков | `0.0` | абсолютная ошибка | verification | [E019](#e019) `/max_absolute_difference` |
| C067 | Все три повторно полученных CSV совпали с ожидаемыми хешами | `true` | логическое значение | verification | [E020](#e020) `/all_regenerated_outputs_match` |
| C068 | SHA256 исходного solution-архива, проверенного в отдельной папке | `"699149dd7f0e84abf13461d5378cf371e701c7da9c3ce49dfcdc92c59cff5aad"` | SHA256 | verification | [E020](#e020) `/archive_sha256` |
| C069 | Сетевой packet capture при проверке не выполнялся | `false` | логическое значение | reported_statement | [E020](#e020) `/offline_execution/network_packet_capture_performed` |
| C070 | Веса двух моделей в выбранной смеси | `[0.5, 0.5]` | веса | design_choice | [E021](#e021) `/weights` |
| C071 | Предельная длина входа объявления E5 | `256` | токенов | design_choice | [E021](#e021) `/dense/item_max_length` |
| C072 | Предельная длина входа запроса E5 | `128` | токенов | design_choice | [E021](#e021) `/dense/query_max_length` |
| C073 | Ревизия загруженной открытой модели по provenance | `"614241f622f53c4eeff9890bdc4f31cfecc418b3"` | git revision | verification | [E024](#e024) `/revision` |
| C074 | Средние метрики совпали с независимым пересчётом таблиц запросов | `true` | логическое значение | verification | [E029](#e029) `/all_checks_passed` |
| C075 | Число успешных проверок независимого аудита | `87` | проверок | verification | [E032](#e032) `/checks_passed` |
| C076 | Общее число проверок независимого аудита | `87` | проверок | verification | [E032](#e032) `/checks_total` |
| C077 | Число текстов train после полного нормализатора NFKC, lower, ё и regex | `73706` | текстов | measurement | [E032](#e032) `/raw_data/train/unique_normalized_texts` |
| C078 | Число скрытых положительных пар, оставшихся в retrieval_train | `0` | пар | verification | [E032](#e032) `/separation/hidden_pair_overlap` |
| C079 | Число холодных текстов, оставшихся в retrieval_train | `0` | текстов | verification | [E032](#e032) `/separation/cold_text_overlap` |
| C080 | Общие целевые item_id между fit+dev и audit при разделении по запросам | `22` | объявлений | measurement | [E032](#e032) `/separation/final_fit_dev_vs_audit_shared_relevant_item_ids` |
| C081 | Независимо пересчитанный Recall@50 выбранной смеси с полным знаменателем релевантности | `0.941111111111111` | доля | verification | [E032](#e032) `/audit/metrics/blend_0.5_0.5/recall@50` |
| C082 | Положительные пары audit, отсутствующие среди гибридных кандидатов | `18` | пар | measurement | [E032](#e032) `/audit/candidate_pool/relevant_items_missing_from_hybrid_pool` |
| C083 | Исключённые audit-запросы с любым пересечением целей с fit или dev | `22` | запросов | measurement | [E032](#e032) `/audit/target_disjoint_sensitivity/removed_queries` |
| C084 | Число audit-запросов в проверке чувствительности без общих целей | `1478` | запросов | measurement | [E032](#e032) `/audit/target_disjoint_sensitivity/remaining_queries` |
| C085 | Пересечение целевых объявлений после исключения запросов | `0` | объявлений | verification | [E032](#e032) `/audit/target_disjoint_sensitivity/remaining_target_item_overlap` |
| C086 | Recall@50 зафиксированной смеси без запросов с общими целями fit или dev | `0.9412494361750112` | доля | measurement | [E032](#e032) `/audit/target_disjoint_sensitivity/macro_recall50` |
| C087 | Recall@50 той же проверки чувствительности с поправкой на пустые фильтры | `0.9429290473343597` | доля | measurement | [E032](#e032) `/audit/target_disjoint_sensitivity/poststratified_recall50` |
| C088 | Новый мини-ноутбук независимо создал CSV с ожидаемым SHA256 | `true` | логическое значение | verification | [E033](#e033) `/exact_csv_hash_match` |
| C089 | Наличие готового answer.csv до независимого запуска мини-ноутбука | `false` | логическое значение | verification | [E033](#e033) `/ready_answer_present_before_execution` |
| C090 | Число фактически выполненных ячеек кода мини-ноутбука | `5` | ячеек | verification | [E033](#e033) `/executed_code_cells` |
| C091 | Длительность выполнения мини-ноутбука в записанном окружении | `5.949` | секунд | measurement | [E033](#e033) `/execution_seconds` |
| C092 | SHA256 CSV независимого запуска мини-ноутбука | `"3ebda47a966aa2920fe9480cb3b4b1602ec91b63f207d89775d981d16995e3c6"` | SHA256 | verification | [E033](#e033) `/generated_csv_sha256` |
| C093 | Размер сохранённой benchmark-таблицы кандидатов | `2604640` | пар запрос и объявление | measurement | [E034](#e034) `/candidate_pool/rows` |
| C094 | Число кандидатов в основном конкурсном CSV | `122600` | пар запрос и объявление | verification | [E034](#e034) `/submission/candidates_total` |
| C095 | Доля 90,4% целей со скрываемой историей задаётся как допущение стресс-проверки, а не измерение скрытого benchmark | `0.904` | доля выбранных целевых объявлений | assumption | [E027](#e027) `round(len(target_ids) * 0.904)` |
| C096 | Равенство CSV после полного пересчёта E5 на CPU и MPS не установлено | `null` | не проверено | limitation | [E018](#e018) `/cpu_mps_equivalence` |

## Состав data dump

Все исходные отчёты JSON, Markdown и Parquet сохранены в data_dump/reports. Параметры обучения, кривые и журналы находятся в data_dump/experiments. Снимок кода находится в data_dump/source_snapshot. E030 перечисляет каждый включённый файл с размером и SHA256.

Полные исходные тексты, dataset.zip, веса, эмбеддинги и таблицы кандидатов не копируются в evidence-архив. Их местоположение и контрольные суммы перечислены в E026. Отдельный solution bundle содержит сохранённые benchmark-признаки и модели для точного повторения CSV.

Самохеширование исключено. E030 не включает собственный хеш, sources.json не ссылается на свой хеш, а архивная опись не включает собственный файл. Финальный SHA256 ZIP выводится отдельно после записи.
