
import pandas as pd

def clean_data(df):
    df.dropna(inplace=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df
