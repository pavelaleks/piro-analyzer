# analysis/trends.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ETH_COL = "Ethnic Group Normalized"

def run(df,
        out_year=RESULTS_DIR / "analysis_4_mentions_by_year.xlsx",
        out_by_eth=RESULTS_DIR / "analysis_4_mentions_by_year_ethnic.xlsx",
        min_year=None,
        max_year=None,
        rolling=3,
        top_ethnic_k=8,
        show_plot=True):
    """
    Mentions trend by publication year (общий) + помесячно по этносам.
    - Заполняет пропущенные годы нулями (не рвём ось времени).
    - Сохраняет 2 файла: общий тренд и сводную по этносам (Year x Ethnic).
    - Опционально рисует график и добавляет скользящее среднее.
    """

    # 1) Готовим колонку Year
    if "Year" not in df.columns:
        raise KeyError("Нет колонки 'Year'.")
    work = df.copy()
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work = work.dropna(subset=["Year"])
    if work.empty:
        print("Нет валидных годов для построения тренда.")
        return
    work["Year"] = work["Year"].astype(int)

    # 2) Общий тренд: считаем и заполняем пропуски нулями
    counts = (work.groupby("Year").size().sort_index())
    year_min, year_max = counts.index.min(), counts.index.max()

    # Диапазон лет
    lo = int(min_year) if min_year is not None else int(year_min)
    hi = int(max_year) if max_year is not None else int(year_max)
    if lo > hi:
        lo, hi = hi, lo

    full_index = pd.Index(range(lo, hi + 1), name="Year")
    trend = counts.reindex(full_index, fill_value=0)

    # 3) Сохраняем общий тренд
    trend_df = trend.rename("Mentions").reset_index()
    trend_df.to_excel(out_year, index=False)
    print(f"Сохранено: {out_year}")

    # 4) Сводная по этносам (если колонка есть)
    by_eth = pd.DataFrame()
    if ETH_COL in work.columns:
        by_eth = (work.groupby(["Year", ETH_COL]).size()
                        .unstack(fill_value=0).sort_index())
        # применяем тот же диапазон лет и 0 для пропусков
        by_eth = by_eth.reindex(full_index, fill_value=0)
        by_eth.to_excel(out_by_eth, index=True)
        print(f"Сохранено: {out_by_eth}")
    else:
        print(f"Колонка '{ETH_COL}' не найдена — пропускаю сводную по этносам.")

    # 5) Консольный вывод
    print("Mentions per publication year:")
    for yr, cnt in trend.items():
        print(f" - {yr}: {int(cnt)}")
    print(f"Years: {lo}–{hi} ({len(trend)} years); total mentions = {int(trend.sum())}\n")

    # Топ лет
    top_years = trend.sort_values(ascending=False).head(10)
    print("Top years by mentions:")
    for yr, cnt in top_years.items():
        print(f" - {yr}: {int(cnt)}")
    print()

    # 6) Графики
    if show_plot:
        # Общий тренд
        x = trend.index.values
        y = trend.values

        plt.figure(figsize=(12, 6))
        plt.plot(x, y, marker="o", label="Mentions")
        if rolling and rolling > 1 and len(trend) >= rolling:
            roll = pd.Series(y, index=x).rolling(rolling, min_periods=1, center=False).mean().values
            plt.plot(x, roll, linestyle="--", label=f"{rolling}-yr rolling avg")
        # подписи над точками (аккуратно)
        y_max = y.max() if len(y) else 0
        shift = max(1, int(y_max * 0.03))
        for xi, yi in zip(x, y):
            if yi > 0:
                plt.text(xi, yi + shift, str(int(yi)), ha="center", va="bottom", fontsize=8)
        plt.xlabel("Year")
        plt.ylabel("Mentions")
        plt.title("Mentions per Publication Year")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # По этносам: линии для Top-K этносов по сумме
        if not by_eth.empty and by_eth.shape[1] > 0:
            totals = by_eth.sum(axis=0).sort_values(ascending=False)
            cols = totals.head(top_ethnic_k).index.tolist()
            sub = by_eth[cols]

            plt.figure(figsize=(12, 6))
            for col in sub.columns:
                plt.plot(sub.index.values, sub[col].values, marker="o", label=col)
            plt.xlabel("Year")
            plt.ylabel("Mentions")
            plt.title(f"Mentions by Year – Top {len(cols)} Ethnic Groups")
            plt.legend()
            plt.tight_layout()
            plt.show()
