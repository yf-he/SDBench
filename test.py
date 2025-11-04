import pyarrow.parquet as pq
import pandas as pd

fp = "/Users/yufei/Desktop/SDBench/shzyk/DiagnosisArena/data/test-00000-of-00001.parquet"
table = pq.read_table(fp)
print("Schema:\n", table.schema)
df = table.to_pandas().head(10)
print(df.to_string(index=False))

