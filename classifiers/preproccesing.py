import pandas as pd

df = pd.read_csv('Dry_Bean_Dataset.csv')

columns_to_remove = ['ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

df = df.drop(columns=columns_to_remove)

df.to_csv('data_set.csv', index=False)