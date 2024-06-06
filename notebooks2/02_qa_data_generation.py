# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()
# COMMAND ----------
import os
from pyspark.sql.functions import rand

from llmsforgood.conf import CATALOG, DATABASE
from llmsforgood.utils.ift_data_prep import prepare_ift_dataset, store_as_mds
from datasets import Dataset

# COMMAND ----------
mds_data_path = f"/Volumes/{CATALOG}/{DATABASE}/data/training/ift/mds/"
jsonl_data_path = f"/Volumes/{CATALOG}/{DATABASE}/data/training/ift/jsonl/"
hf_data_path = f"/Volumes/{CATALOG}/{DATABASE}/data/training/ift/hf_dataset/"

# COMMAND ----------

ift_train_df, ift_val_df = spark.table(f"{CATALOG}.{DATABASE}.qa_dataset").randomSplit(
    [0.99, 0.01]
)
ift_train_df.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{DATABASE}.qa_dataset_train"
)
ift_val_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{DATABASE}.qa_dataset_val")
# COMMAND ----------


ift_completions_train_df = (
    prepare_ift_dataset(
        f"{CATALOG}.{DATABASE}.qa_dataset_train", response_col="good_answer", limit=-1
    )
    .orderBy(rand())
    .repartition(8)
)
ift_completions_val_df = (
    prepare_ift_dataset(
        f"{CATALOG}.{DATABASE}.qa_dataset_val", response_col="good_answer", limit=-1
    )
    .orderBy(rand())
    .repartition(8)
)


display(ift_completions_val_df)  # noqa
# COMMAND ----------
store_as_mds(ift_completions_train_df, os.path.join(mds_data_path, "train"))
store_as_mds(ift_completions_val_df, os.path.join(mds_data_path, "val"))
# COMMAND ----------
ds = Dataset.from_spark(spark.table(f"{CATALOG}.{DATABASE}.qa_dataset_train"))
ds.save_to_disk(hf_data_path)
