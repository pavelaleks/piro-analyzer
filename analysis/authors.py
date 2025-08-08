import matplotlib.pyplot as plt
from collections import Counter

def run(df):
    """Печатает топ-10 авторов и строит график."""
    freq = Counter(df['Author'].dropna().str.strip())
    total = sum(freq.values())
    top10 = freq.most_common(10)

    print("Top 10 authors by mentions:")
    for auth, cnt in top10:
        print(f" - {auth}: {cnt}")
    print(f"Unique authors: {len(freq)}\n")

    if not top10:
        print("No author data found.")
        return

    labels, counts = zip(*top10)
    plt.figure(figsize=(10,6))
    bars = plt.barh(labels, counts)
    plt.xlabel('Count of mentions')
    plt.title('Top 10 Authors')
    for bar, cnt in zip(bars, counts):
        plt.text(cnt + total * 0.005, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va='center', fontsize=8)
    plt.tight_layout()
    plt.show()

def analyze(df):
    """Возвращает статистику по авторам в виде словаря (для экспорта в Excel)."""
    freq = Counter(df['Author'].dropna().str.strip())
    total = sum(freq.values())
    result = {
        'Total Mentions': total,
        'Unique Authors': len(freq)
    }

    if freq:
        top_author, top_count = freq.most_common(1)[0]
        result['Top Author'] = top_author
        result['Top Author Mentions'] = top_count

    return result
