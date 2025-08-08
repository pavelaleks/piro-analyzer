import matplotlib.pyplot as plt

def run(df):
    if 'Year' not in df.columns:
        print("Error: column 'Year' not found.")
        return

    trend = df['Year'].value_counts().sort_index()
    print("Mentions per publication year:")
    for yr, cnt in trend.items():
        print(f" - {yr}: {cnt}")
    print(f"Years: {trend.index.min()}–{trend.index.max()} ({len(trend)} years)\n")

    plt.figure(figsize=(10,5))
    plt.plot(trend.index, trend.values, marker='o')
    for x, y in zip(trend.index, trend.values):
        plt.text(x, y+max(trend.values)*0.01, str(y), ha='center', fontsize=8)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Mentions per Publication Year')
    plt.tight_layout()
    plt.show()
