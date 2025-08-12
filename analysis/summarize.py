# analysis/summary.py
from pathlib import Path
import math
import re
from collections import Counter
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- опционально для тем (gensim). Если нет gensim — блок тем будет пропущен.
try:
    from gensim import corpora, models
    from gensim.models import CoherenceModel
    from gensim.models.phrases import Phrases, Phraser
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

# ---------------- Paths / consts
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ETH_COL = "Ethnic Group Normalized"
AUTH_COL = "Author"
ROLE_CANDS = ["R role", "Role", "R_role", "R"]
TEXT_CANDS = ["Context (±4)", "Context ±4", "Context_EN", "Context", "Excerpt"]
YEAR_COL = "Year"
ID_COL = "Identifier"

# --- Периоды A для сетей
PERIODS_A = [
    ("<=1863", None, 1863),
    ("1864–1895", 1864, 1895),
    ("1896–1914", 1896, 1914),
]
PERIOD_TAG = {"<=1863": "p1", "1864–1895": "p2", "1896–1914": "p3"}

# --- пресеты фильтров для сетей
PRESETS = {
    "exploratory": {"w_min": 1, "j_min": None,  "pmi_min": None, "npmi_min": None},
    "balanced":    {"w_min": 3, "j_min": 0.10,  "pmi_min": 0.0,  "npmi_min": None},
    "strict":      {"w_min": 5, "j_min": 0.20,  "pmi_min": None, "npmi_min": 0.10},
}

# --- токенизация/стоп-слова для тем
AGGRESSIVE_WEB_STOPS = True
REMOVE_TOKENS_WITH_DIGITS = True
REMOVE_SHORT_TOKENS = True
USE_BIGRAMS = True

STOPWORDS_RU = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у",
    "же","вы","за","бы","по","только","ее","мне","было","вот","от","меня","еще","нет","о","из","ему","теперь",
    "когда","даже","ну","вдруг","ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
    "уж","вам","сказал","ведь","там","потом","себя","ничего","ей","может","они","тут","где","есть","надо","ней",
    "для","мы","тебя","их","чем","была","сам","чтоб","без","будто","чего","раз","тоже","себе","под","будет",
    "ж","тогда","кто","этот","того","потому","этого","какой","совсем","ним","здесь","этом","один","почти",
    "мой","тем","чтобы","нее","кажется","разве","другой","сразу","куда","зачем","лишь"
}
STOPWORDS_EN_BASE = {
    "the","and","to","of","in","a","is","that","for","it","as","on","with","are","this","by","an","be","from","at",
    "or","was","which","but","not","have","has","had","were","their","its","they","he","she","we","you","his","her",
    "them","our","one","all","any","more","most","other","some","such","no","nor","only","own","same","so","than",
    "too","very","can","will","just","into","over","also","may","like","these","those","i","me","my","myself","your",
    "yours","yourself","yourselves","him","himself","hers","herself","itself","ours","ourselves","themselves","who",
    "whom","whose","what","when","where","why","how","there","here","then","now","ever","never","always","often",
    "sometimes","once","both","either","neither","each","every","because","although","though","while","until","since",
    "before","after","between","among","within","without","about","above","below","under","across","toward","towards",
    "up","down","out","off","into","onto","again","against","during","through","throughout","per","via"
}
STOPWORDS_EN_WEB = {
    "http","https","www","com","org","net","gov","edu","mil","htm","html","web","website","webpage","page","homepage",
    "click","copy","reserved","ring","msie","microsoft","inc","inc.","ltd","ltd.","co","co.","info","information",
    "recent","recently","welcome","home","help","test","text",
    "zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen",
    "fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty",
    "seventy","eighty","ninety","hundred","thousand","million","billion","trillion",
    "get","got","gets","make","makes","made","begin","beginning","end","ending","find","found","like","likely",
    "use","used","using","based","former","formerly","overall",
    "isn","isn't","aren","aren't","wasn","wasn't","weren","weren't","don","don't","doesn","doesn't","didn","didn't",
    "haven","haven't","hasn","hasn't","wouldn","wouldn't","shouldn","shouldn't","won","won't","can't","cannot",
    "couldn","couldn't","i'd","i'll","i'm","i've","we'd","we'll","we're","we've","he'd","he'll","he's",
    "she'd","she'll","she's","they'd","they'll","they're","they've","what's","that's","there's","who's","where's"
}
STOPWORDS_EN = STOPWORDS_EN_BASE | (STOPWORDS_EN_WEB if AGGRESSIVE_WEB_STOPS else set())
STOPWORDS = STOPWORDS_RU | STOPWORDS_EN
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-']+")

# ---------------- helpers
def _note(ws, text: str, row=1, col=1, freeze_to_row=3):
    try:
        cell = ws.cell(row=row, column=col); cell.value = text
        ws.freeze_panes = f"A{freeze_to_row}"
    except Exception:
        pass

def _sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Sentiment" in df.columns: return df
    if "I" in df.columns:
        s = df["I"].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        out = df.copy(); out["Sentiment"] = pd.to_numeric(s, errors="coerce"); return out
    return df.copy()

def _role_col(df: pd.DataFrame) -> Optional[str]:
    for c in ROLE_CANDS:
        if c in df.columns: return c
    return None

def _split_multi(val: str) -> list[str]:
    seps = [",", ";", "/", "|", " и ", " and "]
    tokens = [val]
    for sep in seps:
        tmp = []
        for t in tokens: tmp.extend([x.strip() for x in str(t).split(sep)])
        tokens = [x for x in tmp if x]
    return tokens

def _tokenize(s: str) -> list[str]:
    toks = []
    for t in TOKEN_RE.findall(s):
        t = t.lower()
        if REMOVE_SHORT_TOKENS and len(t) <= 2: continue
        if REMOVE_TOKENS_WITH_DIGITS and any(ch.isdigit() for ch in t): continue
        if t in STOPWORDS: continue
        toks.append(t)
    return toks

def _pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_CANDS:
        if c in df.columns: return c
    raise KeyError(f"Не найден текстовый столбец из: {TEXT_CANDS}")

def _filter_period(df: pd.DataFrame, lo: int | None, hi: int | None) -> pd.DataFrame:
    if YEAR_COL not in df.columns: return df.copy()
    out = df.copy()
    out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors="coerce")
    out = out.dropna(subset=[YEAR_COL])
    out[YEAR_COL] = out[YEAR_COL].astype(int)
    if lo is not None: out = out[out[YEAR_COL] >= lo]
    if hi is not None: out = out[out[YEAR_COL] <= hi]
    return out

def _edges_from_container(df: pd.DataFrame, container_col: str):
    d = df.dropna(subset=[container_col, ETH_COL]).copy()
    d[container_col] = d[container_col].astype(str).str.strip()
    d[ETH_COL] = d[ETH_COL].astype(str).str.strip()
    groups = d.groupby(container_col)[ETH_COL].apply(lambda s: set(s.dropna()))
    N = len(groups)
    counts = Counter(); edge_weights = Counter()
    for s in groups:
        for a in s: counts[a] += 1
        for a, b in combinations(sorted(s), 2): edge_weights[(a, b)] += 1
    nodes = set(counts.keys())
    return edge_weights, counts, nodes, N

def _edges_metrics_df(edge_weights, counts, N):
    rows = []
    for (a, b), w in edge_weights.items():
        ca, cb = counts.get(a, 0), counts.get(b, 0)
        denom = (ca + cb - w); j = (w / denom) if denom > 0 else 0.0
        p_ab = w / N if N else 0.0; p_a = ca / N if N else 0.0; p_b = cb / N if N else 0.0
        if p_ab > 0 and p_a > 0 and p_b > 0:
            pmi = math.log(p_ab / (p_a * p_b)); npmi = pmi / (-math.log(p_ab))
        else:
            pmi = float("-inf"); npmi = float("-inf")
        rows.append([a, b, int(w), int(ca), int(cb), j, pmi, npmi])
    return pd.DataFrame(rows, columns=["Source","Target","Weight","DocsA","DocsB","Jaccard","PMI","NPMI"])

def _apply_preset(edges_df: pd.DataFrame, preset: dict) -> pd.DataFrame:
    m = edges_df["Weight"] >= preset["w_min"]
    if preset.get("j_min")   is not None: m &= edges_df["Jaccard"] >= preset["j_min"]
    if preset.get("pmi_min") is not None: m &= edges_df["PMI"]     >  preset["pmi_min"]
    if preset.get("npmi_min")is not None: m &= edges_df["NPMI"]    >= preset["npmi_min"]
    return edges_df[m].copy()

def _draw_network(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, title: str, fig_path: Path):
    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["Source"], r["Target"], weight=int(r["Weight"]))
    for n in nodes_df["Node"]:
        if n not in G: G.add_node(n)

    # community
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
        comm_id = {};  [comm_id.setdefault(n, i) for i, cset in enumerate(comms) for n in cset]
        colors = [comm_id.get(n, 0) for n in G.nodes()]
    except Exception:
        colors = [0 for _ in G.nodes()]

    # sizes ~ strength
    str_map = dict(nx.degree(G, weight="weight"))
    smin, smax = (min(str_map.values()) if str_map else 0), (max(str_map.values()) if str_map else 1)
    sizes = [300 + 1200 * ((str_map.get(n,0)-smin) / (smax - smin + 1e-9)) for n in G.nodes()]

    pos = nx.spring_layout(G, weight="weight", seed=42)
    widths = 1.0
    if G.number_of_edges() > 0:
        wts = [d["weight"] for _,_,d in G.edges(data=True)]
        wmin, wmax = min(wts), max(wts)
        widths = [1.0 + 3.0 * (w - wmin) / (wmax - wmin + 1e-9) for w in wts]

    plt.figure(figsize=(10,10))
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap="tab20")
    nx.draw_networkx_labels(G, pos, font_size=8)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): d["weight"] for u,v,d in G.edges(data=True)}, font_size=8)
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close()

# ---------------- MAIN
def run(input_path: str,
        output_path: str | None = None,
        include_network: bool = True,
        include_topics: bool = True,
        doc_mode_topics: str = "document",   # 'document' | 'context' | 'author'
        n_topics: int = 10,
        show_plots: bool = False
        ) -> str:
    """
    Строит максимальный сводный Excel и сохраняет все фигуры в results/figures.
    Возвращает путь к сводному файлу.
    """
    df = pd.read_excel(input_path)

    if output_path is None:
        base = Path(input_path).stem.replace(".xlsx", "")
        output_path = RESULTS_DIR / f"{base}_summary.xlsx"
    else:
        output_path = Path(output_path)

    # ---------- 1) Frequency (F-колонка)
    freq_df = pd.DataFrame()
    if ETH_COL in df.columns:
        freq_series = df[ETH_COL].dropna().astype(str).str.strip()
        freq_df = freq_series.value_counts().reset_index()
        if not freq_df.empty:
            freq_df.columns = [ETH_COL, "Count"]
            # фигура
            labels = freq_df.head(20)[ETH_COL].tolist()
            counts = freq_df.head(20)["Count"].tolist()
            plt.figure(figsize=(12, 6)); plt.bar(labels, counts)
            plt.xticks(rotation=45, ha="right"); plt.title("Top 20 Ethnic Groups by Mentions")
            plt.ylabel("Mentions"); plt.tight_layout()
            plt.savefig(FIG_DIR / "1_frequency_top20.png", dpi=150, bbox_inches="tight")
            if show_plots: plt.show(); plt.close()
        else:
            plt.close()

    # ---------- 2) Sentiment
    df_s = _sentiment_column(df)
    sent_grp_df = pd.DataFrame(); sent_auth_df = pd.DataFrame()
    if ETH_COL in df_s.columns and "Sentiment" in df_s.columns:
        work = df_s.dropna(subset=["Sentiment"]).copy()
        sent_grp_df = (work.groupby(ETH_COL)["Sentiment"]
                           .agg(AvgSentiment="mean", Mentions="size")
                           .sort_values("AvgSentiment", ascending=False).reset_index())

        if not sent_grp_df.empty:
            labels = [f"{g} ({n})" for g,n in zip(sent_grp_df[ETH_COL], sent_grp_df["Mentions"])]
            vals = sent_grp_df["AvgSentiment"].values
            plt.figure(figsize=(12, 6)); plt.bar(labels, vals)
            plt.xticks(rotation=45, ha="right"); plt.ylabel("Average Sentiment")
            plt.title("Average Sentiment by Ethnic Group (N in brackets)"); plt.tight_layout()
            plt.savefig(FIG_DIR / "2_sentiment_by_group.png", dpi=150, bbox_inches="tight")
            if show_plots: plt.show(); plt.close()
        else:
            plt.close()

        if AUTH_COL in work.columns and work[AUTH_COL].notna().any():
            cnts = work.groupby(AUTH_COL)["Sentiment"].size().rename("N")
            means = work.groupby(AUTH_COL)["Sentiment"].mean().rename("AvgSentiment")
            sent_auth_df = (pd.concat([means, cnts], axis=1).query("N >= 3")
                              .sort_values("AvgSentiment", ascending=False).reset_index())
            if not sent_auth_df.empty:
                top = sent_auth_df.head(10)
                labels = top[AUTH_COL].tolist(); vals = top["AvgSentiment"].values
                plt.figure(figsize=(10, 6)); bars = plt.barh(labels, vals)
                plt.xlabel("Average Sentiment"); plt.title("Average Sentiment by Author (Top 10, N≥3)")
                span = float(vals.max()-vals.min()) if len(vals) else 0.0; shift = span*0.02 if span>0 else 0.02
                for bar, v in zip(bars, vals):
                    plt.text(v + shift, bar.get_y()+bar.get_height()/2, f"{v:.2f}", va="center", fontsize=8)
                plt.tight_layout(); plt.savefig(FIG_DIR / "2_sentiment_by_author_top.png", dpi=150, bbox_inches="tight")
                if show_plots: plt.show(); plt.close()
            else:
                plt.close()

    # ---------- 3) Roles crosstab
    ct_counts = pd.DataFrame(); ct_rowpct = pd.DataFrame()
    role_col = _role_col(df)
    if ETH_COL in df.columns and role_col:
        work = df[[ETH_COL, role_col]].dropna(subset=[ETH_COL, role_col]).copy()
        work[ETH_COL] = work[ETH_COL].astype(str).str.strip()
        work[role_col] = work[role_col].astype(str).str.strip().str.lower()
        rows = []
        for _, r in work.iterrows():
            roles = _split_multi(r[role_col]) if r[role_col] else []
            if roles:
                for role in roles:
                    rows.append({ETH_COL: r[ETH_COL], role_col: role})
            else:
                rows.append({ETH_COL: r[ETH_COL], role_col: r[role_col]})
        work = pd.DataFrame(rows)
        ct_counts = pd.crosstab(work[ETH_COL], work[role_col]).sort_index()
        with np.errstate(divide="ignore", invalid="ignore"):
            ct_rowpct = ct_counts.div(ct_counts.sum(axis=1), axis=0).fillna(0)*100
        if not ct_rowpct.empty:
            plt.figure(figsize=(max(8, ct_rowpct.shape[1]*0.6), max(6, ct_rowpct.shape[0]*0.35)))
            plt.imshow(ct_rowpct.values, aspect="auto"); plt.colorbar(label="Row %")
            plt.xticks(range(ct_rowpct.shape[1]), ct_rowpct.columns, rotation=45, ha="right")
            plt.yticks(range(ct_rowpct.shape[0]), ct_rowpct.index)
            plt.title("Roles by Ethnic Group (Row %)"); plt.tight_layout()
            plt.savefig(FIG_DIR / "3_roles_heatmap_rowpct.png", dpi=150, bbox_inches="tight")
            if show_plots: plt.show(); plt.close()
        else:
            plt.close()

    # ---------- 4) Trends
    trend_df = pd.DataFrame(); by_eth = pd.DataFrame()
    if YEAR_COL in df.columns:
        work = df.copy(); work[YEAR_COL] = pd.to_numeric(work[YEAR_COL], errors="coerce")
        work = work.dropna(subset=[YEAR_COL]); work[YEAR_COL] = work[YEAR_COL].astype(int)
        counts = work.groupby(YEAR_COL).size().sort_index()
        if not counts.empty:
            full_index = pd.Index(range(int(counts.index.min()), int(counts.index.max())+1), name=YEAR_COL)
            trend = counts.reindex(full_index, fill_value=0)
            trend_df = trend.rename("Mentions").reset_index()
            x, y = trend.index.values, trend.values
            plt.figure(figsize=(12, 6)); plt.plot(x, y, marker="o", label="Mentions")
            if len(trend) >= 3:
                roll = pd.Series(y, index=x).rolling(3, min_periods=1).mean().values
                plt.plot(x, roll, linestyle="--", label="3-yr rolling avg")
            y_max = y.max() if len(y) else 0; shift = max(1, int(y_max*0.03))
            for xi, yi in zip(x, y):
                if yi > 0: plt.text(xi, yi+shift, str(int(yi)), ha="center", va="bottom", fontsize=8)
            plt.xlabel("Year"); plt.ylabel("Mentions"); plt.legend()
            plt.title("Mentions per Publication Year"); plt.tight_layout()
            plt.savefig(FIG_DIR / "4_trend_by_year.png", dpi=150, bbox_inches="tight")
            if show_plots: plt.show(); plt.close()
        else:
            plt.close()
        if ETH_COL in work.columns:
            by_eth = (work.groupby([YEAR_COL, ETH_COL]).size()
                           .unstack(fill_value=0).sort_index())
            if not by_eth.empty:
                totals = by_eth.sum(axis=0).sort_values(ascending=False)
                cols = totals.head(8).index.tolist(); sub = by_eth[cols]
                plt.figure(figsize=(12, 6))
                for col in sub.columns: plt.plot(sub.index.values, sub[col].values, marker="o", label=col)
                plt.xlabel("Year"); plt.ylabel("Mentions"); plt.legend()
                plt.title("Mentions by Year – Top 8 Ethnic Groups"); plt.tight_layout()
                plt.savefig(FIG_DIR / "4_trend_by_year_top_ethnic.png", dpi=150, bbox_inches="tight")
                if show_plots: plt.show(); plt.close()
            else:
                plt.close()

    # ---------- 5) Networks (document / author; periods A; presets)
    network_overview = []
    network_sheets = []  # (sheet_name, df)
    if include_network and ETH_COL in df.columns:
        for period_name, lo, hi in PERIODS_A:
            dfp = _filter_period(df, lo, hi)
            if dfp.empty: continue
            for mode, cont_col in [("document", ID_COL), ("author", AUTH_COL)]:
                if cont_col not in dfp.columns: continue
                ew, counts, nodes, N = _edges_from_container(dfp, cont_col)
                if not ew:
                    network_overview.append([period_name, mode, N, 0, 0, 0, 0])
                    continue
                edges_all = _edges_metrics_df(ew, counts, N)
                base_tag = f"{PERIOD_TAG[period_name]}_{'doc' if mode=='document' else 'auth'}"
                # обзоры
                network_overview.append([period_name, mode, N, len(nodes), len(edges_all), np.nan, np.nan])
                # запишем полный список (если влезет по размеру листа)
                name_edges_all = f"5_edges_{base_tag}_all"[:31]
                network_sheets.append((name_edges_all, edges_all.sort_values("Weight", ascending=False)))

                # по пресетам
                for pname, flt in PRESETS.items():
                    e = _apply_preset(edges_all, flt)
                    nG = nx.from_pandas_edgelist(e, "Source", "Target", ["Weight"])
                    # nodes_df для этого подграфа
                    degree = dict(nG.degree()); strength = dict(nG.degree(weight="weight"))
                    nodes_df = pd.DataFrame({
                        "Node": list(nG.nodes()),
                        "Degree": [int(degree.get(n,0)) for n in nG.nodes()],
                        "Strength": [float(strength.get(n,0.0)) for n in nG.nodes()],
                        "Containers": [int(counts.get(n,0)) for n in nG.nodes()],
                    }).sort_values(["Strength","Degree"], ascending=False)
                    network_overview.append([period_name, f"{mode}:{pname}", N, len(nG.nodes()), len(nG.edges()), e["Weight"].min() if not e.empty else 0, e["Weight"].max() if not e.empty else 0])
                    # листы (edges/nodes)
                    sheet_e = f"5_{base_tag}_{pname}_e"[:31]
                    sheet_n = f"5_{base_tag}_{pname}_n"[:31]
                    network_sheets.append((sheet_e, e.sort_values("Weight", ascending=False)))
                    network_sheets.append((sheet_n, nodes_df))
                    # картинка
                    if not e.empty:
                        fig_title = f"Co-mention ({mode}, {period_name}, {pname})"
                        fig_name = f"5_{base_tag}_{pname}.png"
                        _draw_network(e, nodes_df, fig_title, FIG_DIR / fig_name)

    network_overview_df = pd.DataFrame(network_overview, columns=["Period","Mode","Containers_N","Nodes","Edges","MinW","MaxW"]) if network_overview else pd.DataFrame()

    # ---------- 6) Topics (gensim)
    topics_df = pd.DataFrame(); docs_topics_df = pd.DataFrame(); eth_topics_df = pd.DataFrame()
    if include_topics and HAS_GENSIM:
        try:
            text_col = _pick_text_col(df)
            # документы = все контексты одного Identifier (или Author/строка)
            if doc_mode_topics == "document" and ID_COL in df.columns:
                groups = df.groupby(ID_COL); doc_ids = []; texts = []; doc2eth = {}
                for doc_id, g in groups:
                    tokens = []
                    for s in g[text_col].dropna().astype(str): tokens.extend(_tokenize(s))
                    if tokens:
                        doc_ids.append(str(doc_id)); texts.append(tokens)
                        doc2eth[str(doc_id)] = set(g.get(ETH_COL, pd.Series(dtype=str)).dropna().astype(str).str.strip())
            elif doc_mode_topics == "author" and AUTH_COL in df.columns:
                groups = df.groupby(AUTH_COL); doc_ids = []; texts = []; doc2eth = {}
                for doc_id, g in groups:
                    tokens = []
                    for s in g[text_col].dropna().astype(str): tokens.extend(_tokenize(s))
                    if tokens:
                        doc_ids.append(str(doc_id)); texts.append(tokens)
                        doc2eth[str(doc_id)] = set(g.get(ETH_COL, pd.Series(dtype=str)).dropna().astype(str).str.strip())
            else:
                series = df[text_col].dropna().astype(str)
                texts = [_tokenize(s) for s in series]; doc_ids = [str(i) for i in series.index]
                doc2eth = {str(i): ({str(df.loc[i, ETH_COL])} if ETH_COL in df.columns and pd.notna(df.loc[i, ETH_COL]) else set()) for i in series.index}

            texts = [t for t in texts if len(t) >= 3]
            if len(texts) >= 5:
                if USE_BIGRAMS:
                    phrases = Phrases(texts, min_count=5, threshold=10.0); bigram = Phraser(phrases)
                    texts = [bigram[t] for t in texts]
                dictionary = corpora.Dictionary(texts)
                dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=20000)
                if len(dictionary) > 0:
                    corpus = [dictionary.doc2bow(t) for t in texts]
                    k = max(2, min(n_topics, len(dictionary), len(texts)//2))
                    lda = models.LdaModel(
                        corpus, num_topics=k, id2word=dictionary,
                        random_state=42, chunksize=2000, passes=10, iterations=200,
                        alpha="auto", eta="auto", eval_every=None
                    )
                    coh = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()
                    # темы
                    rows = []
                    for t_id in range(k):
                        terms = lda.show_topic(t_id, topn=15)
                        rows.append({"Topic": t_id,
                                     "TopTerms": ", ".join(w for w,_ in terms),
                                     "Weights": ", ".join(f"{p:.3f}" for _,p in terms)})
                    topics_df = pd.DataFrame(rows)
                    topics_df.attrs["coherence_c_v"] = coh
                    # doc topics
                    doc_rows = []
                    for doc_id, bow in zip(doc_ids, corpus):
                        dist = sorted(lda.get_document_topics(bow, minimum_probability=0.0), key=lambda x: x[1], reverse=True)
                        row = {"DocID": doc_id, "TopTopic": int(dist[0][0]), "TopProb": float(dist[0][1])}
                        for i,(tid,p) in enumerate(dist[:3],1):
                            row[f"T{i}_id"] = int(tid); row[f"T{i}_p"] = float(p)
                        doc_rows.append(row)
                    docs_topics_df = pd.DataFrame(doc_rows)
                    # eth topics
                    doc_vector = {}
                    for doc_id, bow in zip(doc_ids, corpus):
                        vec = [0.0]*k
                        for tid,p in lda.get_document_topics(bow, minimum_probability=0.0): vec[int(tid)] = float(p)
                        doc_vector[doc_id] = vec
                    eth_rows = []
                    all_eth = set().union(*doc2eth.values()) if doc2eth else set()
                    for eth in all_eth:
                        docs_for_eth = [d for d, es in doc2eth.items() if eth in es]
                        if not docs_for_eth: continue
                        acc = [0.0]*k; n = 0
                        for d in docs_for_eth:
                            v = doc_vector.get(d);
                            if v: acc = [a+b for a,b in zip(acc, v)]; n += 1
                        if n == 0: continue
                        avg = [x/n for x in acc]
                        top = sorted(list(enumerate(avg)), key=lambda x: x[1], reverse=True)[:3]
                        row = {"Ethnic Group Normalized": eth}
                        for i,(tid,p) in enumerate(top,1): row[f"Top{i}_Topic"]=int(tid); row[f"Top{i}_Prob"]=float(p)
                        eth_rows.append(row)
                    if eth_rows:
                        eth_topics_df = pd.DataFrame(eth_rows).sort_values("Top1_Prob", ascending=False)
                    # (не рисуем много картинок по темам, чтобы не плодить десятки PNG)
        except Exception as e:
            print(f"[topics] skipped due to error: {e}")

    # ---------- 7) Authors frequency
    auth_df = pd.DataFrame()
    if AUTH_COL in df.columns:
        s = df[AUTH_COL].dropna().astype(str).str.strip()
        auth_df = s.value_counts().reset_index()
        if not auth_df.empty:
            auth_df.columns = [AUTH_COL, "Count"]
            top = auth_df.head(10)
            labels = top[AUTH_COL].tolist(); counts = top["Count"].tolist()
            plt.figure(figsize=(10, 6)); bars = plt.barh(labels, counts)
            plt.xlabel("Count of mentions"); plt.title(f"Top {len(top)} Authors")
            for bar, cnt in zip(bars, counts):
                plt.text(cnt + max(counts)*0.01, bar.get_y()+bar.get_height()/2, str(int(cnt)), va="center", fontsize=8)
            plt.tight_layout(); plt.savefig(FIG_DIR / "7_authors_top.png", dpi=150, bbox_inches="tight")
            if show_plots: plt.show(); plt.close()
        else:
            plt.close()

    # ---------- WRITE EXCEL
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # README
        readme = (
            "PIRO Summary workbook (auto-generated).\n"
            "Каждый лист сопровождается примечанием в A1 с определениями и допущениями.\n"
            "Фигуры (PNG) сохранены в 'results/figures/'.\n\n"
            "Определения (коротко):\n"
            " - Frequency: value_counts по Ethnic Group Normalized (F-колонка). 1 строка = 1 упоминание.\n"
            " - AvgSentiment: среднее Sentiment по этносу; Mentions = сколько строк вошло в среднее.\n"
            " - Roles: crosstab ролей (counts и row%); многозначные роли сплитятся по , ; / | 'и' 'and'.\n"
            " - TrendByYear: Mentions по годам с заполнением пропусков 0.\n"
            " - Networks: ко-упоминания по Document (Identifier) и Author в периодах A; метрики Weight/Jaccard/PMI/NPMI.\n"
            " - Topics (если gensim доступен): LDA темы; doc_mode=document (все контексты документа склеены).\n"
        )
        pd.DataFrame({"README": [readme]}).to_excel(writer, sheet_name="README", index=False)

        def put(sheet: str, df_: pd.DataFrame, note: str, startrow=2, index=False):
            if df_ is None or df_.empty: return
            df_.to_excel(writer, sheet_name=sheet[:31], startrow=startrow, index=index)
            _note(writer.sheets[sheet[:31]], note)

        put("1_Frequency", freq_df, "Counts of mentions by ethnic group (from column F).")
        put("2_AvgSent_byGroup", sent_grp_df, "Average Sentiment per ethnic group; Mentions shows number of rows used in the mean.")
        put("2_AvgSent_byAuthor", sent_auth_df, "Average Sentiment per author; authors with N<3 are excluded.")
        put("3_Roles_counts", ct_counts, "Crosstab (counts). If a cell had multiple roles, it was split on , ; / | 'и' 'and'.", index=True)
        put("3_Roles_rowpct", ct_rowpct, "Crosstab by row % (each row sums to 100%).", index=True)
        put("4_TrendByYear", trend_df, "Mentions per year; missing years filled with 0.")
        put("4_TrendByYear_Ethnic", by_eth, "Year × Ethnic matrix of mentions (absolute).", index=True)
        put("7_Authors", auth_df, "Mentions per author (raw counts).")

        # Networks
        if network_overview_df is not None and not network_overview_df.empty:
            put("5_Network_Overview", network_overview_df,
                "Overview per period/mode: N containers, nodes/edges after filters. Mode 'document' uses Identifier; 'author' — Author.")
        for sheet_name, data in network_sheets:
            put(sheet_name, data, f"Network data: {sheet_name}. Definitions: Weight=#containers, Jaccard, PMI, NPMI as described in README.")

        # Topics
        if not topics_df.empty:
            put("6_Topics", topics_df, f"LDA topic list; attribute coherence_c_v={topics_df.attrs.get('coherence_c_v'):.3f} (higher≈better).")
        if not docs_topics_df.empty:
            put("6_DocTopics", docs_topics_df, "Per-document dominant topic (TopTopic/TopProb) and Top2/Top3.")
        if not eth_topics_df.empty:
            put("6_Topics_by_Ethnic", eth_topics_df, "For each ethnic group: Top1–Top3 topics aggregated over documents mentioning it.")

    print(f"Summary file written to {output_path}")
    return str(output_path)
