# analysis/sentiment.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run(df, out_groups=RESULTS_DIR / "analysis_2_avg_sentiment_by_group.xlsx",
        out_auth=RESULTS_DIR / "analysis_2_avg_sentiment_by_author.xlsx",
        min_author_n=3,
        show_plot=True):

    # 1) Гарантируем наличие числового Sentiment
    if "Sentiment" not in df.columns:
        if "I" in df.columns:
            s = df["I"].astype(str).str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
            df = df.copy()
            df["Sentiment"] = pd.to_numeric(s, errors="coerce")
        else:
            raise KeyError("Нет колонок 'Sentiment' и 'I' — нечего усреднять.")

    work = df.dropna(subset=["Sentiment"]).copy()
    if work.empty:
        print("Нет данных с числовым Sentiment.")
        return

    # 2) Среднее по этногруппам + количество упоминаний
    if "Ethnic Group Normalized" not in work.columns:
        raise KeyError("Нет колонки 'Ethnic Group Normalized'.")

    grp_stats = (
        work.groupby("Ethnic Group Normalized")["Sentiment"]
        .agg(AvgSentiment="mean", Mentions="size")
        .sort_values("AvgSentiment", ascending=False)
        .reset_index()
    )
    grp_stats.to_excel(out_groups, index=False)
    print(f"Сохранено: {out_groups}")

    # Консольный вывод (Top-10)
    print("Average sentiment per ethnic group (top 10):")
    for _, row in grp_stats.head(10).iterrows():
        print(f" - {row['Ethnic Group Normalized']}: {row['AvgSentiment']:.2f} (N={row['Mentions']})")
    print(f"Groups: {len(grp_stats)}\n")

    # 3) Среднее по авторам + фильтр по количеству наблюдений
    sent_auth = pd.DataFrame(columns=["Author", "AvgSentiment", "N"])
    if "Author" in work.columns and work["Author"].notna().any():
        counts = work.groupby("Author")["Sentiment"].size().rename("N")
        means = work.groupby("Author")["Sentiment"].mean().rename("AvgSentiment")
        sent_auth = (
            pd.concat([means, counts], axis=1)
            .query("N >= @min_author_n")
            .sort_values("AvgSentiment", ascending=False)
        )
        if not sent_auth.empty:
            sent_auth.reset_index().to_excel(out_auth, index=False)
            print(f"Сохранено: {out_auth}")
            print(f"Average sentiment per author (top 10 by AvgSentiment, N≥{min_author_n}):")
            for _, row in sent_auth.head(10).reset_index().iterrows():
                print(f" - {row['Author']} (N={int(row['N'])}): {row['AvgSentiment']:.2f}")
            print(f"Authors (kept): {len(sent_auth)}\n")
        else:
            print(f"По авторам после фильтра N>={min_author_n} данных нет.")
    else:
        print("Колонка 'Author' отсутствует или полностью пустая — пропускаю агрегат по авторам.")

    # 4) Графики
    if show_plot:
        # Группы с количеством
        if len(grp_stats) > 0:
            labels = [
                f"{g} ({n})" for g, n in zip(grp_stats["Ethnic Group Normalized"], grp_stats["Mentions"])
            ]
            vals = grp_stats["AvgSentiment"].values
            plt.figure(figsize=(12, 6))
            plt.bar(labels, vals)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Average Sentiment")
            plt.title("Average Sentiment by Ethnic Group (N in brackets)")
            plt.tight_layout()
            plt.show()

        # Авторы (top 10)
        if not sent_auth.empty:
            top = sent_auth.head(10).reset_index()
            labels = top["Author"].tolist()
            vals = top["AvgSentiment"].values
            plt.figure(figsize=(10, 6))
            bars = plt.barh(labels, vals)
            plt.xlabel("Average Sentiment")
            plt.title(f"Average Sentiment by Author (Top 10, N≥{min_author_n})")
            span = float(vals.max() - vals.min()) if len(vals) else 0.0
            shift = span * 0.02 if span > 0 else 0.02
            for bar, v in zip(bars, vals):
                plt.text(v + shift, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                         va="center", fontsize=8)
            plt.tight_layout()
            plt.show()
