import pyarrow.dataset as ds

dir_name = "software"
dataset = ds.dataset(dir_name, format="csv")
df = dataset.to_table().to_pandas()
df.to_csv(dir_name + ".csv", index=True)