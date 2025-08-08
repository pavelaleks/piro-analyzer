import json
import time
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Пути
BASE_DIR      = Path(__file__).resolve().parents[1]
INPUT_FILE    = BASE_DIR / 'data' / 'results_piro_eng.xlsx'
OUTPUT_FILE   = BASE_DIR / 'data' / 'results_piro_metadata.xlsx'
PROGRESS_FILE = BASE_DIR / 'logs' / 'metadata_progress.jsonl'
LOG_FILE      = BASE_DIR / 'logs' / 'metadata_fetch.log'

# Настройка логирования
LOG_FILE.parent.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Archive.org API

def fetch_metadata_via_api(url: str) -> dict:
    """Получает author и year через API Archive.org"""
    try:
        identifier = url.rstrip('/').split('/')[-1]
        api_url = f'https://archive.org/metadata/{identifier}'
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get('metadata', {})
        # Автор
        author = None
        c = data.get('creator')
        if isinstance(c, list) and c:
            author = c[0]
        elif isinstance(c, str):
            author = c
        # Год
        year = None
        d = data.get('date')
        if d:
            year = str(d).split('-')[0]
        return {'author': author, 'year': year}
    except Exception as e:
        logging.debug(f"API error for {url}: {e}")
        return {'author': None, 'year': None}

# HTML fallback для метаданных

def fetch_metadata_from_archive(url: str) -> dict:
    """Парсит HTML страницы для получения author и year"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        logging.error(f"HTTP error {url}: {e}")
        return {'author': None, 'year': None}
    # 1) JSON-LD
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(tag.string or '{}')
            a = data.get('author')
            author = None
            if isinstance(a, list) and a:
                author = a[0].get('name')
            elif isinstance(a, dict):
                author = a.get('name')
            dp = data.get('datePublished')
            year = dp.split('-')[0] if dp else None
            if author or year:
                return {'author': author, 'year': year}
        except Exception:
            continue
    # 2) Meta теги
    author_meta = soup.find('meta', attrs={'name': 'creator'})
    year_meta   = soup.find('meta', attrs={'name': 'datePublished'})
    author = author_meta['content'].strip() if author_meta and author_meta.get('content') else None
    year_raw = year_meta['content'].strip() if year_meta and year_meta.get('content') else None
    year = year_raw.split('-')[0] if year_raw else None
    if author or year:
        return {'author': author, 'year': year}
    # 3) OG/DC
    og = soup.find('meta', property='og:updated_time')
    if og and og.get('content'):
        return {'author': None, 'year': og['content'].split('-')[0]}
    dc = soup.find('meta', attrs={'name': 'DC.date'})
    if dc and dc.get('content'):
        return {'author': None, 'year': dc['content'].split('-')[0]}
    # 4) DT/DD
    author = None; year = None
    for dt in soup.find_all('dt'):
        label = dt.get_text(strip=True).lower()
        dd = dt.find_next_sibling('dd')
        if not dd:
            continue
        text = dd.get_text(strip=True)
        if label == 'by':
            author = text
        elif 'publication date' in label:
            year = text.split()[0]
    return {'author': author, 'year': year}

# Работа с прогрессом

def load_progress() -> set:
    """Читает PROGRESS_FILE и возвращает set обработанных индексов"""
    done = set()
    if PROGRESS_FILE.exists():
        for line in PROGRESS_FILE.read_text(encoding='utf-8').splitlines():
            try:
                done.add(json.loads(line)['idx'])
            except json.JSONDecodeError:
                continue
    return done


def append_progress(idx: int, author, year):
    """Добавляет одну запись в JSONL-прогресс"""
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        json.dump({'idx': idx, 'author': author, 'year': year}, f)
        f.write('\n')


def apply_progress_to_df(df: pd.DataFrame):
    """Проставляет ранее сохранённые author/year в df"""
    if not PROGRESS_FILE.exists():
        return
    for line in PROGRESS_FILE.read_text(encoding='utf-8').splitlines():
        try:
            rec = json.loads(line)
            df.at[rec['idx'], 'Author'] = rec.get('author')
            df.at[rec['idx'], 'Year']   = rec.get('year')
        except json.JSONDecodeError:
            continue

# Обработать только пропуски
def process_missing(df: pd.DataFrame):
    missing_idx = df[df['Author'].isna() | df['Year'].isna()].index
    total = len(missing_idx)
    for count, idx in enumerate(missing_idx, start=1):
        url = df.at[idx, 'File Link']
        print(f"[{count}/{total}] Processing row {idx}")
        try:
            meta = fetch_metadata_via_api(url)
            if not meta['year']:
                meta = fetch_metadata_from_archive(url)
            print(f"[DEBUG] idx={idx}, author={meta['author']}, year={meta['year']}")
            df.at[idx, 'Author'] = meta['author']
            df.at[idx, 'Year']   = meta['year']
            append_progress(idx, meta['author'], meta['year'])
        except Exception as e:
            logging.exception(f"Error on row {idx}: {e}")
        time.sleep(1)

# Основной блок
if __name__ == '__main__':
    # Загрузка и инициализация
    df = pd.read_excel(INPUT_FILE)
    for col in ('Author', 'Year'):
        if col not in df.columns:
            df[col] = None
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Применить ранее сохранённый прогресс
    apply_progress_to_df(df)

    # Обработать только пропущенные
    process_missing(df)

    # Диагностика и сохранение
    total = len(df)
    print(f"Authors fetched: {df['Author'].notnull().sum()} / {total}")
    print(f"Years fetched:   {df['Year'].notnull().sum()} / {total}")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")