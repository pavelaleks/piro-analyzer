# analysis/topics.py
from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ETH_COL   = "Ethnic Group Normalized"
ID_COL    = "Identifier"
AUTHOR_COL= "Author"
TEXT_CANDS= ["Context (±4)", "Context ±4", "Context_EN", "Context", "Excerpt"]

# ====== НАСТРОЙКИ ЧИСТКИ ======
AGGRESSIVE_WEB_STOPS = True          # включает дополнительные web/служебные стоп-слова (по твоему списку)
REMOVE_TOKENS_WITH_DIGITS = True     # убираем токены с цифрами (в т.ч. 153-9024)
REMOVE_SHORT_TOKENS = True           # убираем токены длиной <= 2
USE_BIGRAMS = True                   # биграммы gensim (можно выключить, если корпус мал)

# Базовые стоп-слова (RU+EN, безопасные)
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
# Твои дополнительные web/служебные стоп-слова (собраны и нормализованы)
STOPWORDS_EN_WEB = {
    # инфраструктура/домены/протоколы
    "http","https","www","com","org","net","gov","edu","mil","htm","html","web","website","webpage","page","homepage",
    "click","copy","reserved","ring","msie","microsoft",
    # «мусорные» служебные/офисные
    "inc","inc.","ltd","ltd.","co","co.","info","information","recent","recently","welcome","home","help","test","text",
    # числовые слова
    "zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen",
    "fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty",
    "ninety","hundred","thousand","million","billion","trillion",
    # частотные глаголы/частицы из твоего списка, которые малоинформативны для тем
    "get","got","gets","make","makes","made","begin","beginning","end","ending","find","found","like","likely",
    "use","used","using","used","use","based","former","formerly","overall",
    # отрицания/сокращения форм
    "isn","isn't","aren","aren't","wasn","wasn't","weren","weren't","don","don't","doesn","doesn't","didn","didn't",
    "haven","haven't","hasn","hasn't","wouldn","wouldn't","shouldn","shouldn't","won","won't","can't","cannot","couldn",
    "couldn't","i'd","i'll","i'm","i've","we'd","we'll","we're","we've","he'd","he'll","he's","she'd","she'll","she's",
    "they'd","they'll","they're","they've","what's","that's","there's","who's","where's"
}
# Собираем итоговый набор
STOPWORDS_EN = STOPWORDS_EN_BASE | (STOPWORDS_EN_WEB if AGGRESSIVE_WEB_STOPS else set())
STOPWORDS = STOPWORDS_RU | STOPWORDS_EN

# Разрешаем буквы (латиница/кириллица), дефис и апострофы
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-']+")

def _tokenize(s: str) -> list[str]:
    toks = []
    for t in TOKEN_RE.findall(s):
        t = t.lower()
        if REMOVE_SHORT_TOKENS and len(t) <= 2:
            continue
        if REMOVE_TOKENS_WITH_DIGITS and any(ch.isdigit() for ch in t):
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks

def _pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_CANDS:
        if c in df.columns:
            return c
    raise KeyError(f"Не найден текстовый столбец из: {TEXT_CANDS}")

def _build_docs(df: pd.DataFrame, doc_mode: str, text_col: str):
    # как и раньше: 'document' (Identifier) по умолчанию, затем 'author', затем 'context'
    if doc_mode == "document" and ID_COL in df.columns:
        groups = df.groupby(ID_COL)
        doc_ids, texts, doc2eth = [], [], {}
        for doc_id, g in groups:
            tokens = []
            for s in g[text_col].dropna().astype(str):
                tokens.extend(_tokenize(s))
            if tokens:
                doc_ids.append(str(doc_id))
                texts.append(tokens)
                doc2eth[str(doc_id)] = set(g.get(ETH_COL, pd.Series(dtype=str)).dropna().astype(str).str.strip())
        return texts, doc_ids, doc2eth
    if doc_mode == "author" and AUTHOR_COL in df.columns:
        groups = df.groupby(AUTHOR_COL)
        doc_ids, texts, doc2eth = [], [], {}
        for doc_id, g in groups:
            tokens = []
            for s in g[text_col].dropna().astype(str):
                tokens.extend(_tokenize(s))
            if tokens:
                doc_ids.append(str(doc_id))
                texts.append(tokens)
                doc2eth[str(doc_id)] = set(g.get(ETH_COL, pd.Series(dtype=str)).dropna().astype(str).str.strip())
        return texts, doc_ids, doc2eth
    # fallback: каждая строка — документ
    series = df[text_col].dropna().astype(str)
    texts = [_tokenize(s) for s in series]
    doc_ids = [str(i) for i in series.index]
    doc2eth = {str(i): ({str(df.loc[i, ETH_COL])} if ETH_COL in df.columns and pd.notna(df.loc[i, ETH_COL]) else set())
               for i in series.index}
    return texts, doc_ids, doc2eth

def run(df: pd.DataFrame, n_topics: int = 10, doc_mode: str = "document",
        use_bigrams: bool = USE_BIGRAMS,
        out_topics=RESULTS_DIR / "analysis_6_topics.xlsx",
        out_docs=RESULTS_DIR / "analysis_6_doc_topics.xlsx",
        out_eth=RESULTS_DIR / "analysis_6_topics_by_ethnic.xlsx",
        show_print: bool = True):

    text_col = _pick_text_col(df)
    texts, doc_ids, doc2eth = _build_docs(df, doc_mode, text_col)

    # фильтруем слишком короткие тексты
    texts = [t for t in texts if len(t) >= 3]
    if len(texts) < 5:
        print("Слишком мало текстов для LDA после чистки. Попробуй doc_mode='document' или ослабь фильтры.")
        return

    # биграммы
    if use_bigrams:
        phrases = Phrases(texts, min_count=5, threshold=10.0)
        bigram = Phraser(phrases)
        texts = [bigram[t] for t in texts]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=20000)
    if len(dictionary) == 0:
        print("После фильтрации словарь пуст. Уменьши жесткость фильтров или отключи bigrams.")
        return

    corpus = [dictionary.doc2bow(t) for t in texts]
    k = max(2, min(n_topics, len(dictionary), len(texts)//2))

    lda = models.LdaModel(
        corpus, num_topics=k, id2word=dictionary,
        random_state=42, chunksize=2000, passes=10, iterations=200,
        alpha="auto", eta="auto", eval_every=None
    )

    coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")
    coh = coherence_model.get_coherence()

    # Темы
    rows = []
    for t_id in range(k):
        terms = lda.show_topic(t_id, topn=15)
        rows.append({"Topic": t_id,
                     "TopTerms": ", ".join(w for w, _ in terms),
                     "Weights": ", ".join(f"{p:.3f}" for _, p in terms)})
    topics_df = pd.DataFrame(rows)
    topics_df.attrs["coherence_c_v"] = coh
    topics_df.to_excel(out_topics, index=False)
    print(f"Сохранено: {out_topics} (coherence c_v={coh:.3f}, k={k}, mode={doc_mode})")

    # Темы по документам
    doc_topics = []
    for doc_id, bow in zip(doc_ids, corpus):
        dist = sorted(lda.get_document_topics(bow, minimum_probability=0.0), key=lambda x: x[1], reverse=True)
        row = {"DocID": doc_id, "TopTopic": int(dist[0][0]), "TopProb": float(dist[0][1])}
        for i, (tid, p) in enumerate(dist[:3], 1):
            row[f"T{i}_id"] = int(tid); row[f"T{i}_p"] = float(p)
        doc_topics.append(row)
    pd.DataFrame(doc_topics).to_excel(out_docs, index=False)
    print(f"Сохранено: {out_docs}")

    # Темы по этносам (среднее по документам, где встречается этнос)
    doc_vector = {}
    for doc_id, bow in zip(doc_ids, corpus):
        vec = [0.0]*k
        for tid, p in lda.get_document_topics(bow, minimum_probability=0.0):
            vec[int(tid)] = float(p)
        doc_vector[doc_id] = vec

    eth_rows = []
    # собираем doc_id -> этносы
    for eth, docs in defaultdict(list, {e: [d for d, es in doc2eth.items() if e in es] for e in set().union(*doc2eth.values())}).items():
        if not docs: continue
        acc = [0.0]*k
        n = 0
        for d in docs:
            v = doc_vector.get(d)
            if v: acc = [a+b for a,b in zip(acc, v)]; n += 1
        if n == 0: continue
        avg = [x/n for x in acc]
        top = sorted(list(enumerate(avg)), key=lambda x: x[1], reverse=True)[:3]
        row = {"Ethnic Group Normalized": eth}
        for i, (tid, p) in enumerate(top, 1):
            row[f"Top{i}_Topic"] = int(tid); row[f"Top{i}_Prob"] = float(p)
        eth_rows.append(row)
    if eth_rows:
        pd.DataFrame(eth_rows).sort_values("Top1_Prob", ascending=False).to_excel(out_eth, index=False)
        print(f"Сохранено: {out_eth}")

    if show_print:
        print(f"LDA Topics (k={k}, c_v={coh:.3f}, mode={doc_mode}):")
        for _, r in topics_df.iterrows():
            print(f"Topic {int(r['Topic'])}: {r['TopTerms']}")
        print()
