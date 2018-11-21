import pandas as pd

def load_csv(path):
    return pd.read_csv(path, index_col=0)
