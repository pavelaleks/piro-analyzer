# analysis/comention_network.py
from pathlib import Path
from collections import Counter
from itertools import combinations
import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# колонки
ETH_COL   = "Ethnic Group Normalized"
ID_COL    = "Identifier"
AUTHOR_COL= "Author"
YEAR_COL  = "Year"

# периоды (вариант A)
PERIODS_A = [
    ("<=1863", None, 1863),
    ("1864–1895", 1864, 1895),
    ("1896–1914", 1896, 1914),
]

# пресеты фильтров
PRESETS = {
    "exploratory": {"w_min": 1, "j_min": None,  "pmi_min": None, "npmi_min": None},
    "balanced":    {"w_min": 3, "j_min": 0.10,  "pmi_min": 0.0,  "npmi_min": None},
    "strict":      {"w_min": 5, "j_min": 0.20,  "pmi_min": None, "npmi_min": 0.10},
}

DRAW_TOP_NODES = 30
SEED = 42


def _filter_period(df: pd.DataFrame, lo: int | None, hi: int | None) -> pd.DataFrame:
    if YEAR_COL not in df.columns:
        return df.copy()
    out = df.copy()
    out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors="coerce")
    out = out.dropna(subset=[YEAR_COL])
    if lo is not None:
        out = out[out[YEAR_COL] >= lo]
    if hi is not None:
        out = out[out[YEAR_COL] <= hi]
    return out


def _edges_from_container(df: pd.DataFrame, container_col: str):
    """Строим веса рёбер по контейнеру (документ или автор)."""
    d = df.dropna(subset=[container_col, ETH_COL]).copy()
    d[container_col] = d[container_col].astype(str).str.strip()
    d[ETH_COL] = d[ETH_COL].astype(str).str.strip()

    # множества этносов внутри каждого контейнера
    groups = d.groupby(container_col)[ETH_COL].apply(lambda s: set(s.dropna()))
    N = len(groups)  # число контейнеров в срезе

    # сколько контейнеров у каждого этноса
    counts = Counter()
    edge_weights = Counter()
    for s in groups:
        for a in s:
            counts[a] += 1
        for a, b in combinations(sorted(s), 2):
            edge_weights[(a, b)] += 1

    nodes = set(counts.keys())
    return edge_weights, counts, nodes, N


def _edges_metrics_df(edge_weights, counts, N):
    rows = []
    for (a, b), w in edge_weights.items():
        ca, cb = counts.get(a, 0), counts.get(b, 0)
        denom = (ca + cb - w)
        j = (w / denom) if denom > 0 else 0.0

        p_ab = w / N if N else 0.0
        p_a  = ca / N if N else 0.0
        p_b  = cb / N if N else 0.0
        if p_ab > 0 and p_a > 0 and p_b > 0:
            pmi = math.log(p_ab / (p_a * p_b))
            npmi = pmi / (-math.log(p_ab))
        else:
            pmi = float("-inf")
            npmi = float("-inf")

        rows.append([a, b, int(w), int(ca), int(cb), j, pmi, npmi])
    return pd.DataFrame(rows, columns=["Source", "Target", "Weight", "DocsA", "DocsB", "Jaccard", "PMI", "NPMI"])


def _apply_preset(edges_df: pd.DataFrame, preset: dict) -> pd.DataFrame:
    m = edges_df["Weight"] >= preset["w_min"]
    if preset.get("j_min") is not None:
        m &= edges_df["Jaccard"] >= preset["j_min"]
    if preset.get("pmi_min") is not None:
        m &= edges_df["PMI"] > preset["pmi_min"]
    if preset.get("npmi_min") is not None:
        m &= edges_df["NPMI"] >= preset["npmi_min"]
    return edges_df[m].copy()


def _nodes_df(G: nx.Graph, counts_per_node: Counter) -> pd.DataFrame:
    degree   = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    return pd.DataFrame({
        "Node": list(G.nodes()),
        "Containers": [int(counts_per_node.get(n, 0)) for n in G.nodes()],  # сколько доков/авторов упоминали этнос
        "Degree": [int(degree.get(n, 0)) for n in G.nodes()],
        "Strength": [float(strength.get(n, 0.0)) for n in G.nodes()],
    }).sort_values(["Strength", "Degree"], ascending=False)


def _draw_graph(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, title: str):
    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["Source"], r["Target"], weight=int(r["Weight"]))
    for n in nodes_df["Node"]:
        if n not in G:
            G.add_node(n)

    # раскраска по сообществам (если получится)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
        comm_id = {}
        for i, cset in enumerate(comms):
            for n in cset: comm_id[n] = i
        colors = [comm_id.get(n, 0) for n in G.nodes()]
    except Exception:
        colors = [0 for _ in G.nodes()]

    # размеры узлов ~ strength
    str_map = dict(zip(nodes_df["Node"], nodes_df["Strength"]))
    if str_map:
        smin, smax = min(str_map.values()), max(str_map.values())
    else:
        smin, smax = 0.0, 1.0
    sizes = [300 + 1200 * ((str_map.get(n, 0.0) - smin) / (smax - smin + 1e-9)) for n in G.nodes()]

    pos = nx.spring_layout(G, weight="weight", seed=SEED)

    # толщина рёбер ~ вес
    if G.number_of_edges() > 0:
        wts = [d["weight"] for _, _, d in G.edges(data=True)]
        wmin, wmax = min(wts), max(wts)
        widths = [1.0 + 3.0 * (w - wmin) / (wmax - wmin + 1e-9) for w in wts]
    else:
        widths = 1.0

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap="tab20")
    nx.draw_networkx_labels(G, pos, font_size=8)

    if G.number_of_edges() > 0:
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run(df: pd.DataFrame,
        containers=("document", "author"),
        periods=PERIODS_A,
        show_plot=True):
    """
    Строит сети по периодам A и двум уровням контейнера:
    - document: Identifier
    - author:   Author
    Сохраняет edges/nodes c метриками + рисует три графа (exploratory/balanced/strict).
    """
    if ETH_COL not in df.columns:
        raise KeyError(f"Нет колонки '{ETH_COL}'")

    for period_name, lo, hi in periods:
        dfp = _filter_period(df, lo, hi)
        if dfp.empty:
            print(f"[{period_name}] нет данных.")
            continue

        for mode in containers:
            if mode == "document":
                if ID_COL not in dfp.columns:
                    print(f"[{period_name}][document] нет колонки '{ID_COL}', пропускаю.")
                    continue
                cont_col = ID_COL
            elif mode == "author":
                if AUTHOR_COL not in dfp.columns:
                    print(f"[{period_name}][author] нет колонки '{AUTHOR_COL}', пропускаю.")
                    continue
                cont_col = AUTHOR_COL
            else:
                raise ValueError("containers must be 'document' or 'author'")

            ew, counts, nodes, N = _edges_from_container(dfp, cont_col)
            if not ew:
                print(f"[{period_name}][{mode}] ко-упоминаний нет.")
                continue

            edges_all = _edges_metrics_df(ew, counts, N)
            nodes_all = _nodes_df(
                nx.from_pandas_edgelist(edges_all, "Source", "Target", ["Weight"]),
                counts
            )

            base = f"analysis_5_{period_name.replace('–','-').replace('<=','le')}_{mode}"
            edges_all.to_csv(RESULTS_DIR / f"{base}_edges_all.csv", index=False, encoding="utf-8")
            nodes_all.to_csv(RESULTS_DIR / f"{base}_nodes_all.csv", index=False, encoding="utf-8")
            print(f"Сохранено: {RESULTS_DIR / (base + '_edges_all.csv')}")
            print(f"Сохранено: {RESULTS_DIR / (base + '_nodes_all.csv')}")

            # три пресета
            for pname, flt in PRESETS.items():
                e = _apply_preset(edges_all, flt)
                if e.empty:
                    print(f"[{period_name}][{mode}][{pname}] после фильтра рёбер нет.")
                    continue
                n = _nodes_df(nx.from_pandas_edgelist(e, "Source", "Target", ["Weight"]), counts)

                e.to_csv(RESULTS_DIR / f"{base}_edges_{pname}.csv", index=False, encoding="utf-8")
                n.to_csv(RESULTS_DIR / f"{base}_nodes_{pname}.csv", index=False, encoding="utf-8")
                print(f"Сохранено: {RESULTS_DIR / (base + f'_edges_{pname}.csv')}")
                print(f"Сохранено: {RESULTS_DIR / (base + f'_nodes_{pname}.csv')}")

                if show_plot:
                    _draw_graph(
                        e,
                        n,
                        title=f"Co-mention ({mode}, {period_name}, {pname})"
                    )
