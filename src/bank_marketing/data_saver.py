from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from bank_marketing.config import ProjectConfig

class BaseDataSaver(ABC):
    """Abstract base class for saving data."""

    @abstractmethod
    def save(self, df: pd.DataFrame, name: str):
        pass

class DatabricksSaver(BaseDataSaver):
    """Saves data to Databricks."""

    def __init__(self, config: ProjectConfig, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.databricks_catalog = f"{config.catalog_name}.{config.schema_name}"

    def save(self, df: pd.DataFrame, name: str):
        """Save the dataset to Databricks Delta Lake."""
        df_with_ts = self.spark.createDataFrame(df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        df_with_ts.write.mode("append").saveAsTable(f"{self.databricks_catalog}.{name}")

        self.spark.sql(
            f"ALTER TABLE {self.databricks_catalog}.{name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        logger.info(f"Dataset {name} saved to {self.databricks_catalog}.{name}")
