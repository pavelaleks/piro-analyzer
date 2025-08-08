import pandas as pd

def run(df):
    if 'R' not in df.columns:
        print("Error: column 'R' not found.")
        return

    ct = pd.crosstab(df['Ethnic Group Normalized'], df['R'])
    print("Roles crosstab (first 10 rows):")
    print(ct.head(10).to_string())
    print(f"\nTotal ethnic groups: {ct.shape[0]}, role categories: {ct.shape[1]}\n")

    ct.to_csv('roles_crosstab.csv')
    print("Full crosstab saved to roles_crosstab.csv")
