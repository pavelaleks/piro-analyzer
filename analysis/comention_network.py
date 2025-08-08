import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def run(df):
    G = nx.Graph()
    for ctx, grp in df.groupby('Context (±4)')['Ethnic Group Normalized']:
        unique = set(grp.dropna())
        for a in unique:
            for b in unique:
                if a != b:
                    G.add_edge(a.title(), b.title())

    print(f"Network: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    deg = Counter(dict(G.degree()))
    top5 = deg.most_common(5)
    print("Top 5 by degree:")
    for node, d in top5:
        print(f" - {node}: {d}")
    print()

    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
    plt.title("Co-mention Network")
    plt.tight_layout()
    plt.show()
