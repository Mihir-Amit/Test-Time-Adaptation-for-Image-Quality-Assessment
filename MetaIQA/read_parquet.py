import pandas as pd

# Read a Parquet file
df = pd.read_parquet('only_rank_loss.parquet')

# Display the first few rows of the DataFrame
print(df.iloc[-20:])

