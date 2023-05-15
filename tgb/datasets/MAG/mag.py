import pandas as pd


if __name__ == "__main__":
    df = pd.read_parquet("nodes.parquet/nodes.parquet", engine="pyarrow")
    data_top = df.head()

    print(data_top)
