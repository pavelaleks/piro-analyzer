import matplotlib.pyplot as plt

def run(df):
    # Сентимент по этногруппам
    sent_grp = df.groupby('Ethnic Group Normalized')['Sentiment'].mean().sort_values()
    # Сентимент по авторам
    sent_auth = df.groupby('Author')['Sentiment'].mean().sort_values()

    print("Average sentiment per ethnic group:")
    for grp, val in sent_grp.items():
        print(f" - {grp.title()}: {val:.2f}")
    print(f"Groups: {len(sent_grp)}\n")

    print("Average sentiment per author (top 10):")
    for auth, val in sent_auth.head(10).items():
        print(f" - {auth}: {val:.2f}")
    print(f"Authors: {len(sent_auth)}\n")

    # График групп
    labels = [g.title() for g in sent_grp.index]
    vals = sent_grp.values
    plt.figure(figsize=(10, 5))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Sentiment')
    plt.title('Average Sentiment by Ethnic Group')
    plt.tight_layout()
    plt.show()

    # График авторов
    labels = sent_auth.head(10).index
    vals   = sent_auth.head(10).values
    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, vals)
    plt.xlabel('Average Sentiment')
    plt.title('Average Sentiment by Author (Top 10)')
    span = max(vals) - min(vals)
    for bar, v in zip(bars, vals):
        plt.text(v + span*0.01, bar.get_y()+bar.get_height()/2, f"{v:.2f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.show()
