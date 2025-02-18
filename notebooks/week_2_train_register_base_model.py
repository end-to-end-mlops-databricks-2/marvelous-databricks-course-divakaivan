# Databricks notebook source
from loguru import logger
import mlflow
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.base_model import BaseModel

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yaml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
base_model = BaseModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
base_model.load_data()
base_model.prepare_pipeline()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
base_model.train()
base_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name_base], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/hgbc-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
base_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
base_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
base_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.{config.test_set_name}").limit(10)

X_test = test_set.drop(config.target, "update_timestamp_utc").toPandas()

predictions_df = base_model.load_latest_model_and_predict(X_test)
logger.info(predictions_df)
# COMMAND ----------