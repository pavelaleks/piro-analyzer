from gensim import corpora, models
import re

def run(df, n_topics=10):
    texts = df['Context (±4)'].dropna().astype(str).apply(lambda s: re.findall(r"\w+", s.lower())).tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=10, random_state=42)

    print(f"LDA Topics (n_topics={n_topics}):")
    for idx, topic in lda.print_topics(num_words=6):
        print(f"Topic {idx}: {topic}")
    print()
