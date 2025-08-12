# frequency.py
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path  # <-- добавили

SPLIT_MULTI = False
SEPARATORS = [",", ";", "/", "&", " и ", " and "]

def _split_if_needed(s: str) -> list[str]:
    if not SPLIT_MULTI:
        return [s]
    tokens = [s]
    for sep in SEPARATORS:
        new_tokens = []
        for t in tokens:
            new_tokens.extend([x.strip() for x in t.split(sep)])
        tokens = [x for x in new_tokens if x]
    return tokens

def run(df, output_path="results/analysis_1_frequency_of_mentions.xlsx", top_k=10, show_plot=True):
    # --- гарантируем, что каталог существует ---
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    col = "Ethnic Group Normalized"
    if col not in df.columns:
        raise KeyError(f"Колонка '{col}' не найдена в данных")

    series = df[col].dropna().astype(str).str.strip()

    if SPLIT_MULTI:
        all_items = []
        for val in series:
            all_items.extend(_split_if_needed(val))
        freq = pd.Series(Counter(all_items)).sort_values(ascending=False)
    else:
        freq = series.value_counts(dropna=False)

    out_df = freq.reset_index()
    out_df.columns = [col, "Count"]
    out_df.to_excel(out_path, index=False)
    print(f"Сохранено: {out_path}")

    total = int(out_df["Count"].sum())
    print(f"Total mentions: {total}, unique groups: {len(out_df)}")
    print("Top {}:".format(min(top_k, len(out_df))))
    for _, row in out_df.head(top_k).iterrows():
        print(f" - {row[col]}: {row['Count']}")
    print()

    if show_plot and len(out_df) > 0:
        labels = out_df[col].tolist()
        counts = out_df["Count"].tolist()
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, counts)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Frequency by '{col}' (Total: {total})")
        for bar, cnt in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, cnt, str(cnt), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.show()
