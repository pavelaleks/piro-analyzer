# PIRO Analyzer

**PIRO Analyzer** — инструмент для анализа историко‑этнографических травелогов по PIRO‑модели (*People – Image – Representation – Occasion*). Проект на Python, ориентирован на исследование репрезентаций этнических групп в путешественнической литературе Российской империи (≈ 1850–1917).

## Возможности
- Частоты упоминаний этнонимов (по колонке **F = Ethnic Group Normalized**).
- Средний сентимент по этносам и авторам.
- Перекрёстная таблица «роли × этносы» (counts и доли по строке).
- Динамика упоминаний по годам (общая и по этносам).
- Сети совместных упоминаний (по документам и по авторам) с метриками **Weight/Jaccard/PMI/NPMI** в трёх пресетах (exploratory/balanced/strict) и по периодам **A**.
- Тематическое моделирование (LDA) по документам/авторам/контекстам.
- Сводный Excel‑отчёт + набор PNG‑графиков (если используете модуль `analysis/summary.py`).

---

## Структура проекта

```
piro-analyzer/
├─ analysis/                # модули анализа (frequency, sentiment, roles, trends, comention_network, topics, authors, summary)
├─ data/                    # входные Excel (результаты PIRO-разметки)
├─ results/                 # все выходные таблицы и summary (создаётся автоматически)
│  └─ figures/              # PNG-графики (создаётся автоматически)
├─ main.py                  # интерактивное меню для запуска анализаторов
├─ requirements.txt         # зависимости
└─ README.md                # этот файл
```

---

## Установка

```bash
git clone https://github.com/pavelaleks/piro-analyzer.git
cd piro-analyzer
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Быстрый старт

1) Поместите основной файл данных в `data/`, по умолчанию:  
   `data/results_piro_metadata.xlsx`

2) Запустите меню:
```bash
python main.py
```
В меню:
```
1. Frequency of mentions
2. Average sentiment
3. Roles crosstab
4. Mentions trend by year
5. Co-mention network
6. Topic modeling
7. Frequency of authors
8. All analyses
0. Exit
```

3) Результаты появятся в `results/` и графики — в `results/figures/`.

> **Важно (Windows/Excel):** если какой‑то `.xlsx` открыт в Excel, сохранение в него завершится ошибкой `Permission denied`. Закройте файл и повторите запуск, либо используйте «безопасное сохранение» в модулях (создаётся копия с таймстампом).

---

## Требования к данным (минимум)

Желательно наличие следующих колонок (регистрозависимые заголовки):

- **Ethnic Group Normalized** *(обяз.)* — этноним (используется в 1, 2, 3, 4, 5, 6).  
- **Identifier** — ID документа (5; агрегирование 6 при `doc_mode=document`).  
- **Year** — год публикации (4; фильтры эпох для 5).  
- **Author** — автор (2, 5 в режиме author, 7).  
- **I** или **Sentiment** — числовой сентимент; если `Sentiment` отсутствует, число извлекается из `I`.  
- **R / Role / R_role / R role** — роль для кросстаба (3).  
- **Context (±4)** *(или Context/Excerpt)* — текст для тем (6).

---

## Где что сохраняется

- **Таблицы** — в `results/` (по анализаторам) и/или на листах сводника (`results/summary_book.xlsx`, если включён).
- **Графики** — в `results/figures/`:
  - `1_frequency_top20.png`
  - `2_sentiment_by_group.png`, `2_sentiment_by_author_top.png`
  - `3_roles_heatmap_rowpct.png`
  - `4_trend_by_year.png`, `4_trend_by_year_top_ethnic.png`
  - `5_*.png` — сети (все периоды/режимы/пресеты)
  - `7_authors_top.png`

> Существующие результаты не удаляются. PNG‑графики с одинаковыми именами перезаписываются при новом запуске.

---

## Как читать результаты (анализаторы 1–7)

### 0) Общие определения
- **Упоминание** — одна строка данных с непустой **F = Ethnic Group Normalized**.  
- **Документ** — значение в `Identifier`.  
- **Периоды A** (для сетей): **≤1863 (p1)**, **1864–1895 (p2)**, **1896–1914 (p3)**.  
- **Sentiment** — из `Sentiment`, иначе извлекается из `I` (число со знаком).  
- **Графики** — PNG в `results/figures/`. Таблицы — по отдельным анализаторам и/или в своднике.

---

### 1) Frequency of mentions (частоты упоминаний)

**Лист:** `1_Frequency`  
**Файл:** `results/analysis_1_frequency_of_mentions.xlsx`

**Колонки**
- `Ethnic Group Normalized` — этноним (как в колонке F).
- `Count` — сколько строк содержит этот этноним.

**График:** `figures/1_frequency_top20.png` — топ‑20 по абсолютам (масштаб без нормировок).

**Замечание:** сравнивайте частоты вместе с ролями/годами; «популярные» группы доминируют.

---

### 2) Average sentiment (средний сентимент)

**Листы:** `2_AvgSent_byGroup`, `2_AvgSent_byAuthor`  
**Файлы:**  
`results/analysis_2_avg_sentiment_by_group.xlsx`,  
`results/analysis_2_avg_sentiment_by_author.xlsx`

**By Group — колонки**
- `Ethnic Group Normalized` — этнос.
- `AvgSentiment` — средний `Sentiment` по этносу.
- `Mentions` — сколько строк вошло в среднее (контроль доверия).

**By Author — колонки**
- `Author` — (после мягкой нормализации).
- `AvgSentiment` — средний `Sentiment` по автору.
- `N` — сколько строк учтено (обычно оставляем `N ≥ 3`).

**Графики**  
`figures/2_sentiment_by_group.png` — столбцы = `AvgSentiment`, подпись `(N)`.  
`figures/2_sentiment_by_author_top.png` — топ‑10 авторов по среднему.

**Замечание:** маленький `N` ⇒ среднее ненадёжно.

---

### 3) Roles crosstab (роли × этносы)

**Листы:** `3_Roles_counts`, `3_Roles_rowpct`  
**Файлы:**  
`results/analysis_3_roles_crosstab_counts.xlsx`,  
`results/analysis_3_roles_crosstab_rowpct.xlsx`,  
`results/analysis_3_roles_mentions_per_ethnic.xlsx`

**Counts**  
Строки = этносы; столбцы = роли (берутся из `R role`/`Role`/`R_role`/`R`).  
Если в ячейке несколько ролей, значения делятся по разделителям: `, ; / | "и" "and"`.

**Row%**  
Те же строки/столбцы, но каждая строка нормирована до 100% (структура, а не масштаб).

**График:** `figures/3_roles_heatmap_rowpct.png` — теплокарта долей по строке.

**Замечание:** для интерпретации держите рядом абсолюты (`counts`) и `mentions_per_ethnic` (N по этносам).

---

### 4) Mentions trend by year (тренды по годам)

**Листы:** `4_TrendByYear`, `4_TrendByYear_Ethnic`  
**Файлы:**  
`results/analysis_4_mentions_by_year.xlsx`,  
`results/analysis_4_mentions_by_year_ethnic.xlsx`

**`4_TrendByYear`**  
`Year` — непрерывный диапазон `[min; max]` (пропуски = 0), `Mentions` — количество строк/год.

**`4_TrendByYear_Ethnic`**  
Матрица `Year × Ethnic Group Normalized` (абсолюты).

**Графики**  
`figures/4_trend_by_year.png` — общий тренд + 3‑летнее скользящее среднее.  
`figures/4_trend_by_year_top_ethnic.png` — топ‑8 этносов по сумме.

**Замечание:** при сравнении периодов полезна нормировка на объём корпуса (если доступна).

---

### 5) Co-mention network (сети совместных упоминаний)

**Лист:** `5_Network_Overview` + листы `5_edges_{pX}_{doc|auth}_{preset}`, `5_{pX}_{doc|auth}_{preset}_n`  
**Картинки:** `figures/5_{pX}_{doc|auth}_{exploratory|balanced|strict}.png`

**Режимы (containers)**
- **document** — контейнер = `Identifier` (пара встречается в одном документе).
- **author** — контейнер = `Author` (пара встречается у одного автора).

**Периоды A**  
`≤1863` (p1), `1864–1895` (p2), `1896–1914` (p3).

**Пресеты фильтров**
- *exploratory*: `Weight ≥ 1`
- *balanced*: `Weight ≥ 3` и (`Jaccard ≥ 0.10` или `PMI > 0`)
- *strict*: `Weight ≥ 5` и (`Jaccard ≥ 0.20` или `NPMI ≥ 0.10`)

**Edges — колонки**
- `Source`, `Target` — этносы.  
- `Weight` — в скольких контейнерах пара встречалась.  
- `DocsA`, `DocsB` — в скольких контейнерах встречались A и B по отдельности.  
- `Jaccard = Weight / (DocsA + DocsB − Weight)`  
- `PMI = ln(Weight * N / (DocsA * DocsB))`, где `N` — число контейнеров.  
- `NPMI` — нормированная PMI в [−1; 1].

**Nodes — колонки**
- `Node` — этнос.  
- `Degree` — число соседей (невзвешенно).  
- `Strength` — сумма весов рёбер.  
- `Containers` — в скольких контейнерах встречался узел.

**Как читать рисунки сети**  
Размер узла ~ `Strength`, цвет — сообщество (по модульности), подписи рёбер — `Weight`.  
Документный режим показывает «сосуществование в одном источнике», авторский — «карты внимания» авторов.

**Замечание:** ориентируйтесь на `balanced/strict` для устойчивых связей; сравнивайте *doc* vs *author*.

---

### 6) Topic modeling (LDA)

**Листы:** `6_Topics`, `6_DocTopics`, `6_Topics_by_Ethnic`  
**Файлы:** `results/analysis_6_topics.xlsx`, `analysis_6_doc_topics.xlsx`, `analysis_6_topics_by_ethnic.xlsx`

**Метод / текст**  
По умолчанию `doc_mode = "document"`: все контексты одного `Identifier` объединяются в документ.  
Токенизация: нижний регистр; RU/EN + web стоп‑слова; токены с цифрами и длиной ≤2 удаляются; опционально биграммы.

**`6_Topics`**  
- `Topic` (0…k−1), `TopTerms` (15 слов), `Weights` (веса слов).  
- Атрибут файла: **coherence c_v** (ориентир качества; ~0.4–0.6 для коротких текстов приемлемо).

**`6_DocTopics`**  
- `DocID`, `TopTopic`, `TopProb`, `T1_id/T1_p`, `T2_id/T2_p`, `T3_id/T3_p`.

**`6_Topics_by_Ethnic`**  
- `Ethnic Group Normalized`, `Top1/2/3` темы и вероятности — усреднение по документам, где этнос встречается.

**Рекомендации**  
Давайте интерпретируемые имена тем по `TopTerms`. Для жёсткого отбора документов по теме используйте `TopProb ≥ 0.6–0.7`.

---

### 7) Frequency of authors (авторы)

**Лист:** `7_Authors`  
**Файл:** `results/analysis_7_frequency_of_authors.xlsx`

**Колонки**
- `Author` — автор (мягкая нормализация пробелов/знаков).  
- `Mentions` — упоминаний (строк).  
- `Documents` — уникальных документов автора (если есть `Identifier`).  
- `MentionsPerDoc` — упоминаний на документ.  
- `FirstYear`, `LastYear` — диапазон лет (если есть `Year`).  
- `AvgSentiment` — средний `Sentiment` (если есть).

**График:** `figures/7_authors_top.png` — топ‑10 по `Mentions` (подпись может содержать `docs=…`).

**Замечание:** ориентируйтесь не только на `Mentions`, но и на `Documents`, чтобы не переоценивать один «толстый» источник.

---

## Сводный отчёт (summary, опционально)

Если используете `analysis/summary.py`, можно собрать один файл со всеми ключевыми таблицами и комментариями на листах:

- **Файл:** `results/summary_book.xlsx` (или с таймстампом).  
- **Листы:** `README`, `1_Frequency`, `2_*`, `3_*`, `4_*`, `5_*`, `6_*`, `7_Authors`.  
- **Графики:** в `results/figures/`.

Пример вызова из `main.py` (если добавляете пункт меню самостоятельно):
```python
from analysis.summary import run as summary_run
summary_run(
    args.input,
    output_path="results/summary_book.xlsx",
    include_network=True,
    include_topics=True,
    doc_mode_topics="document",
    n_topics=10,
    show_plots=False
)
```

---

## Параметры (по умолчанию, важно для воспроизводимости)

- **Сети:** периоды A; режимы `document`/`author`; пресеты фильтра — см. раздел 5.  
- **Темы (LDA):** `doc_mode=document`, `n_topics=10` c безопасным ограничением `k ≤ min(n_topics, |словарь|, ⌊docs/2⌋)`; биграммы включены; агрессивные EN web‑стоп‑слова включены; токены с цифрами удаляются.  
- **Сентимент:** строки без числового значения исключаются из средних (по этносам и авторам).

---

## Ограничения и проверки качества

- Для средних и долей **всегда** контролируйте `N` (`Mentions`, `N`, `Containers`).  
- В `Roles_rowpct` высокая доля при маленьком `N` может быть артефактом — сверяйтесь с абсолютами.  
- Сети на уровне документов — это **сосуществование в одном источнике**, а не в одном абзаце; авторские сети отображают «карты внимания».  
- Для тем ориентируйтесь на **coherence c_v** и список `TopTerms`; при необходимости уменьшайте число тем.

---

## Частые проблемы

- **`PermissionError: [Errno 13] Permission denied` при записи XLSX** — файл открыт в Excel. Закройте его и перезапустите «8. All analyses»; либо используйте безопасное сохранение (файл будет записан с таймстампом).  
- **`ModuleNotFoundError: from analysis ...`** — запускайте из корня проекта: `python main.py`.  
- **Пустые графики/таблицы** — проверьте наличие обязательных колонок и непустых значений; для LDA корпус мог стать слишком мал после чистки.

---

## Лицензия и цитирование

## Как цитировать

Alekseev Pavel (2025). PIRO Analyzer (Version 1.0.0) [Software]. GitHub.
https://github.com/pavelaleks/piro-analyzer

Если вы используете сводный отчёт, пожалуйста укажите:
summary_book.xlsx, версия 1.0.0, выгружено 2025-08-12.
