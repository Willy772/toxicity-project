import pandas as pd
from pathlib import Path
from .config import CSV_PATH, N_ROWS

def load_df(csv_path: Path | str = CSV_PATH, n_rows: int = N_ROWS) -> pd.DataFrame:
    df_all = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return df_all.head(n_rows).copy()
