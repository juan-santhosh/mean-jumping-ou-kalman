import pandas as pd

MINUTES_PER_MONTH = 60 * 24 * 30

def load_data(filepath: str, months: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="|", index_col=0) 
    df = df.iloc[-(MINUTES_PER_MONTH * months):].reset_index()

    return df