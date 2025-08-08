import matplotlib.pyplot as plt
from collections import Counter
import re
import unicodedata

# Маппинг этнонимов (оставьте свой полный словарь)
NORMALIZATION_MAP = {
    'samoyed': 'samoyeds', 'samoyedes': 'samoyeds',
    'baribats': 'buryats', 'buriats': 'buryats', 'buriatz': 'buryats', 'buriates': 'buryats', 'buryats': 'buryats',
    'kirghiz': 'kyrgyz', 'kirgiz': 'kyrgyz', 'kyrgiz': 'kyrgyz', 'kyrgyz': 'kyrgyz',
    'kalmuk': 'kalmuks', 'kalmuks': 'kalmuks',
    'tungus': 'tungus', 'tunguses': 'tungus',
    'yakut': 'yakuts', 'yakuts': 'yakuts', 'jakut': 'yakuts', 'jakuts': 'yakuts',
    'ostyak': 'khanty', 'ostyaks': 'khanty', 'khanty': 'khanty',
    'vogul': 'mansi', 'voguls': 'mansi', 'mansi': 'mansi',
    'tatars': 'tatars', 'bashkirs': 'bashkirs',
}

def clean_key(name: str) -> str:
    nk = unicodedata.normalize('NFKD', name)
    return re.sub(r'[^A-Za-z]', '', nk).lower()

def normalize_group(name: str) -> str:
    key = clean_key(name)
    if key in NORMALIZATION_MAP:
        canon = NORMALIZATION_MAP[key]
    elif key.startswith('yakut'):
        canon = 'yakuts'
    else:
        canon = key
    return canon.title()

def run(df):
    raw = df['Ethnic Group Normalized'].dropna().astype(str)
    groups = raw.apply(normalize_group)

    freq = Counter(groups)
    total = sum(freq.values())
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Текстовый вывод
    print(f"Total mentions: {total}, unique groups: {len(items)}")
    print("Top 10 ethnic groups:")
    for grp, cnt in items[:10]:
        print(f" - {grp}: {cnt}")
    print()

    # График
    labels, counts = zip(*items)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Frequency of mentions (Total: {total})")
    for bar, cnt in zip(bars, counts):
        plt.text(bar.get_x()+bar.get_width()/2, cnt + total*0.005, str(cnt), ha='center', fontsize=8)
    plt.tight_layout()
    plt.show()
