import pandas as pd

# Этот скрипт собирает основные сводные данные и сохраняет их в один Excel-файл

def run(input_path: str, output_path: str = None):
    df = pd.read_excel(input_path)

    # Определяем выходной файл
    if output_path is None:
        output_path = input_path.replace('.xlsx', '_summary.xlsx')

    # 1. Frequency of mentions
    freq = df['Ethnic Group Normalized'].value_counts().rename_axis('Ethnic Group').reset_index(name='Count')

    # 2. Average sentiment by group
    avg_sent_group = df.groupby('Ethnic Group Normalized')['Sentiment']\
                       .mean()\
                       .reset_index()\
                       .rename(columns={'Ethnic Group Normalized': 'Ethnic Group', 'Sentiment': 'Avg Sentiment'})

    # 3. Roles crosstab
    roles_ct = pd.crosstab(df['Ethnic Group Normalized'], df['R'])
    roles_ct.index.name = 'Ethnic Group'

    # 4. Mentions trend by publication year
    trend = df['Year'].value_counts().sort_index().rename_axis('Year').reset_index(name='Count')

    # 5. Frequency of authors
    auth_freq = df['Author'].value_counts().rename_axis('Author').reset_index(name='Count')

    # Сохраняем все в Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        freq.to_excel(writer, sheet_name='Frequency', index=False)
        avg_sent_group.to_excel(writer, sheet_name='AvgSentiment', index=False)
        roles_ct.to_excel(writer, sheet_name='RolesCrosstab')
        trend.to_excel(writer, sheet_name='TrendByYear', index=False)
        auth_freq.to_excel(writer, sheet_name='AuthorFrequency', index=False)

    print(f"Summary file written to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate summary Excel workbook")
    parser.add_argument('--input', default='data/results_piro_metadata.xlsx',
                        help='Path to enriched Excel file')
    parser.add_argument('--output', default=None,
                        help='Path to summary Excel file (optional)')
    args = parser.parse_args()
    run(args.input, args.output)
