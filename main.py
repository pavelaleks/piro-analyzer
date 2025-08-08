import pandas as pd
import argparse
from analysis.frequency import run as freq_run
from analysis.sentiment import run as sent_run
from analysis.roles import run as roles_run
from analysis.trends import run as trends_run
from analysis.comention_network import run as net_run
from analysis.topics import run as topics_run
from analysis.authors import run as auth_run  # новый модуль

def load_data(path):
    return pd.read_excel(path)

def preprocess(df):
    # Убираем пустые ключевые поля и создаём копию для безопасных присвоений
    df = df.dropna(subset=['Ethnic Group Normalized', 'I', 'R', 'Author', 'Year']).copy()
    # Year уже числовой после metadata.py
    df.loc[:, 'Year'] = df['Year'].astype(int)
    # Sentiment парсится как раньше
    df.loc[:, 'Sentiment'] = df['I'].str.extract(r'([\-\d\.]+)')[0].astype(float)
    return df

def select_analyses():
    options = [
        ('1', 'Frequency of mentions'),
        ('2', 'Average sentiment'),
        ('3', 'Roles crosstab'),
        ('4', 'Mentions trend by year'),
        ('5', 'Co-mention network'),
        ('6', 'Topic modeling'),
        ('7', 'Frequency of authors'),
        ('8', 'All analyses'),
        ('0', 'Exit'),
    ]
    print('\nSelect analyses to run:')
    for code, desc in options:
        print(f"  {code}. {desc}")
    return input('Enter number: ').strip()

def main():
    parser = argparse.ArgumentParser(description="PIRO travelogue analysis")
    parser.add_argument('--input',
        default='data/results_piro_metadata.xlsx',
        help='Path to the enriched Excel data file')
    args = parser.parse_args()

    df = load_data(args.input)
    df = preprocess(df)

    while True:
        choice = select_analyses()
        if choice == '0':
            break
        elif choice == '1':
            freq_run(df)
        elif choice == '2':
            sent_run(df)
        elif choice == '3':
            roles_run(df)
        elif choice == '4':
            trends_run(df)
        elif choice == '5':
            net_run(df)
        elif choice == '6':
            topics_run(df, n_topics=10)
        elif choice == '7':
            auth_run(df)
        elif choice == '8':
            freq_run(df); sent_run(df); roles_run(df)
            trends_run(df); net_run(df); topics_run(df, n_topics=10)
            auth_run(df)
        else:
            print('Invalid choice')

if __name__ == '__main__':
    main()