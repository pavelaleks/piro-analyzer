# analysis/authors.py
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUTH_COL = "Author"
ID_COL   = "Identifier"
YEAR_COL = "Year"

# При желании можно заполнить ручными синонимами:
AUTHOR_NORMALIZATION_MAP = {
    # пример:
    # "Landsdell, Henry": "Lansdell, Henry",
}

def _normalize_author(a: str) -> str:
    """Мягкая нормализация имён: пробелы, запятые, хвостовые точки.
       Без жёсткого title-case, чтобы не ломать Mc-/дефисы/инициалы."""
    a = a.replace("\u00A0", " ")             # неразрывные пробелы
    a = " ".join(a.split())                   # лишние пробелы внутри
    a = re.sub(r"\s*,\s*", ", ", a)           # пробелы вокруг запятых
    a = re.sub(r"[.\s]+$", "", a)             # хвостовые точки/пробелы
    canon = AUTHOR_NORMALIZATION_MAP.get(a, a)
    return canon

def run(df: pd.DataFrame,
        out_path=RESULTS_DIR / "analysis_7_frequency_of_authors.xlsx",
        min_author_n: int = 1,
        show_plot: bool = True):
    if AUTH_COL not in df.columns:
        raise KeyError(f"Нет колонки '{AUTH_COL}'")

    work = df[[c for c in [AUTH_COL, ID_COL, YEAR_COL, "Sentiment"] if c in df.columns]].copy()
    work[AUTH_COL] = work[AUTH_COL].astype(str).str.strip()
    work = work[work[AUTH_COL].str.len() > 0]

    # Нормализуем «мягко»
    work["AuthorNorm"] = work[AUTH_COL].apply(_normalize_author)

    # Базовые счёты
    mentions = work.groupby("AuthorNorm").size().rename("Mentions")
    # Сколько разных документов у автора (если есть Identifier)
    docs = (work.groupby("AuthorNorm")[ID_COL].nunique().rename("Documents")) if ID_COL in work.columns else None
    # Диапазон лет (если есть Year)
    if YEAR_COL in work.columns:
        years = work.dropna(subset=[YEAR_COL]).copy()
        years[YEAR_COL] = pd.to_numeric(years[YEAR_COL], errors="coerce")
        year_min = years.groupby("AuthorNorm")[YEAR_COL].min().rename("FirstYear")
        year_max = years.groupby("AuthorNorm")[YEAR_COL].max().rename("LastYear")
    else:
        year_min = year_max = None
    # Средний сентимент (если есть)
    if "Sentiment" in work.columns:
        sent = (work.dropna(subset=["Sentiment"])
                    .groupby("AuthorNorm")["Sentiment"].mean()
                    .rename("AvgSentiment"))
    else:
        sent = None

    parts = [mentions]
    if docs is not None: parts.append(docs)
    if year_min is not None: parts += [year_min, year_max]
    if sent is not None: parts.append(sent)

    out = pd.concat(parts, axis=1).fillna({"Documents": 0})
    if "Documents" in out.columns:
        out["MentionsPerDoc"] = (out["Mentions"] / out["Documents"]).replace([pd.NA, float("inf")], 0)

    # Порог по минимальному числу упоминаний
    if min_author_n > 1:
        out = out[out["Mentions"] >= min_author_n]

    out = out.sort_values(["Mentions", "Documents"], ascending=False).reset_index().rename(columns={"AuthorNorm": "Author"})
    out.to_excel(out_path, index=False)
    print(f"Сохранено: {out_path}")
    print(f"Всего авторов: {len(out)}; суммарные упоминания: {int(out['Mentions'].sum())}")

    # Консольный топ-10
    top10 = out.head(10)
    print("Top 10 authors by mentions:")
    for _, r in top10.iterrows():
        docs_info = f", docs={int(r['Documents'])}" if "Documents" in out.columns else ""
        span_info = f", {int(r['FirstYear'])}–{int(r['LastYear'])}" if "FirstYear" in out.columns else ""
        print(f" - {r['Author']}: {int(r['Mentions'])}{docs_info}{span_info}")

    # График
    if show_plot and not top10.empty:
        labels = top10["Author"].tolist()
        counts = top10["Mentions"].astype(int).tolist()
        plt.figure(figsize=(10, 6))
        bars = plt.barh(labels, counts)
        plt.xlabel("Count of mentions")
        plt.title("Top 10 Authors")
        # подписи: N и количество документов, если есть
        for bar, cnt, _, row in zip(bars, counts, range(len(top10)), top10.to_dict("records")):
            extra = f" (docs={int(row['Documents'])})" if "Documents" in row else ""
            plt.text(cnt + max(counts)*0.01, bar.get_y()+bar.get_height()/2,
                     f"{cnt}{extra}", va="center", fontsize=8)
        plt.tight_layout()
        plt.show()
