# Databricks notebook source

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor
from bank_marketing.data_saver import DatabricksSaver
from bank_marketing.utils import create_schema

# COMMAND ----------

logger.info("Loading configuration")
config = ProjectConfig.from_yaml(config_path="../project_config.yaml")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
logger.info("Creating Spark session")
spark = SparkSession.builder.getOrCreate()

schema = create_schema(config.raw_data_schema)
logger.info("Reading raw data")
df = spark.read.csv(
    "/Volumes/mlops_dev/diva4eto/bank_marketing_volume/bank_marketing.csv", header=True, schema=schema, sep=";"
).toPandas()

# COMMAND ----------

data_processor = DataProcessor(df, config)

logger.info("Preprocessing data")
data_processor.clean_column_names()
data_processor.preprocess()

# COMMAND ----------

logger.info("Splitting data")
train_set, test_set = data_processor.split_data()
logger.info(f"Training set shape: {train_set.shape}")
logger.info(f"Test set shape: {test_set.shape}")

# COMMAND ----------

logger.info("Saving data to catalog")
db_saver = DatabricksSaver(config, spark)
db_saver.save(train_set, config.train_set_name)
db_saver.save(test_set, config.test_set_name)

# COMMAND ----------
