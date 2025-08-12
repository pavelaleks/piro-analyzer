# analysis/roles.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# если в одной ячейке несколько ролей — разбивать?
SPLIT_MULTI = True
SEPARATORS = [",", ";", "/", "|", " и ", " and "]

# минимум упоминаний этноса для попадания в теплокарту/таблицы с фильтром
MIN_MENTIONS = 5

ROLE_COL_CANDIDATES = ["R role", "Role", "R_role", "R"]
ETH_COL = "Ethnic Group Normalized"

def _pick_role_col(df: pd.DataFrame) -> str:
    for c in ROLE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"Не нашли колонку роли из вариантов: {ROLE_COL_CANDIDATES}")

def _split_multi(val: str) -> list[str]:
    tokens = [val]
    for sep in SEPARATORS:
        tmp = []
        for t in tokens:
            tmp.extend([x.strip() for x in t.split(sep)])
        tokens = [x for x in tmp if x]
    return tokens

def run(df: pd.DataFrame,
        out_counts=RESULTS_DIR / "analysis_3_roles_crosstab_counts.xlsx",
        out_rowpct=RESULTS_DIR / "analysis_3_roles_crosstab_rowpct.xlsx",
        out_counts_filtered=RESULTS_DIR / "analysis_3_roles_crosstab_counts_filtered.xlsx",
        out_rowpct_filtered=RESULTS_DIR / "analysis_3_roles_crosstab_rowpct_filtered.xlsx",
        out_mentions=RESULTS_DIR / "analysis_3_roles_mentions_per_ethnic.xlsx",
        show_heatmap=True):

    # проверки
    if ETH_COL not in df.columns:
        raise KeyError(f"Нет колонки '{ETH_COL}'")
    role_col = _pick_role_col(df)

    work = df[[ETH_COL, role_col]].dropna(subset=[ETH_COL, role_col]).copy()
    work[ETH_COL] = work[ETH_COL].astype(str).str.strip()
    # роли в нижний регистр для консистентности (подписи оставим как есть в столбцах сводной)
    work[role_col] = work[role_col].astype(str).str.strip().str.lower()

    # раздуваем строки при множественных ролях
    if SPLIT_MULTI:
        rows = []
        for _, r in work.iterrows():
            roles = _split_multi(r[role_col]) if r[role_col] else []
            if roles:
                for role in roles:
                    rows.append({ETH_COL: r[ETH_COL], role_col: role})
            else:
                rows.append({ETH_COL: r[ETH_COL], role_col: r[role_col]})
        work = pd.DataFrame(rows)

    # сводная: абсолютные количества
    ct_counts = pd.crosstab(work[ETH_COL], work[role_col]).sort_index()
    # сумма по этносам
    mentions = ct_counts.sum(axis=1).rename("Mentions").to_frame()
    mentions.sort_values("Mentions", ascending=False).to_excel(out_mentions, index=True)

    # доли по строкам (в процентах)
    with np.errstate(divide="ignore", invalid="ignore"):
        ct_rowpct = ct_counts.div(ct_counts.sum(axis=1), axis=0).fillna(0) * 100

    # полные таблицы
    ct_counts.to_excel(out_counts)
    ct_rowpct.to_excel(out_rowpct)
    print(f"Сохранено: {out_counts}")
    print(f"Сохранено: {out_rowpct}")
    print(f"Сохранено: {out_mentions}")

    # отфильтрованные по MIN_MENTIONS (для устойчивой визуализации/сводки)
    mask = mentions["Mentions"] >= MIN_MENTIONS
    kept_eth = mentions.index[mask]
    ct_counts_f = ct_counts.loc[kept_eth] if len(kept_eth) else pd.DataFrame(columns=ct_counts.columns)
    ct_rowpct_f = ct_rowpct.loc[kept_eth] if len(kept_eth) else pd.DataFrame(columns=ct_rowpct.columns)

    if not ct_counts_f.empty:
        ct_counts_f.to_excel(out_counts_filtered)
        ct_rowpct_f.to_excel(out_rowpct_filtered)
        print(f"Сохранено: {out_counts_filtered}")
        print(f"Сохранено: {out_rowpct_filtered}")
    else:
        print(f"После порога MIN_MENTIONS={MIN_MENTIONS} этносов не осталось — пропускаю сохранение фильтрованных таблиц.")

    # консоль: кусок
    print("Roles crosstab (counts, first 10 rows):")
    print(ct_counts.head(10).to_string())
    print(f"\nTotal ethnic groups: {ct_counts.shape[0]}, role categories: {ct_counts.shape[1]}\n")

    # теплокарта по долям для отфильтрованных (если есть), иначе для полных
    if show_heatmap:
        mat = ct_rowpct_f if not ct_rowpct_f.empty else ct_rowpct
        ment = mentions.loc[mat.index] if not mat.empty else mentions
        if not mat.empty:
            # подписи вида "Yakuts (N=47)"
            ylabels = [f"{eth} (N={int(ment.loc[eth, 'Mentions'])})" for eth in mat.index]

            plt.figure(figsize=(max(8, mat.shape[1] * 0.6), max(6, mat.shape[0] * 0.35)))
            plt.imshow(mat.values, aspect="auto")
            plt.colorbar(label="Row %")
            plt.xticks(ticks=range(mat.shape[1]), labels=mat.columns, rotation=45, ha="right")
            plt.yticks(ticks=range(mat.shape[0]), labels=ylabels)
            title_suffix = f" (N≥{MIN_MENTIONS})" if mat is ct_rowpct_f else ""
            plt.title(f"Roles by Ethnic Group (Row %){title_suffix}")
            plt.tight_layout()
            plt.show()
        else:
            print("Нет данных для теплокарты.")