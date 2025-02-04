from typing import Optional

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from bank_marketing.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df
        self.config = config

    def clean_column_names(self):
        """Preprocess the data stored in self.df"""

        self.df.rename(
            columns={
                "age": "age",
                "balance": "avg_annual_balance",
                "duration": "last_contact_duration_in_sec",
                "previous": "n_contacts_prev_campaign",
                "campaign": "n_contacts_cur_campaign",
                "pdays": "days_since_prev_contact",
                "job": "job",
                "marital": "marital_status",
                "education": "education_level",
                "default": "has_default",
                "housing": "has_housing_loan",
                "loan": "has_personal_loan",
                "contact": "contact_type",
                "day": "last_contact_day_of_week",
                "month": "last_contact_month",
                "poutcome": "outcome_prev_campaign",
                "y": "has_subscribed",
            },
            inplace=True,
        )

    def preprocess(self):
        logger.info("Processing numerical features")
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        logger.info("Processing numerical features")

        logger.info("Processing categorical features")
        cat_features = self.config.cat_features
        for col in cat_features:
            self.df[col] = self.df[col].astype("category").cat.codes
        logger.info("Processing categorical features")

        logger.info("Processing target")
        target = self.config.target
        self.df[target] = self.df[target].map({"no": 0, "yes": 1})
        logger.info("Processing target")

        needed_cols = num_features + cat_features + [target]
        self.df = self.df[needed_cols]

    def split_data(self, test_size: Optional[float] = 0.2, random_state: Optional[int] = 42):
        """Split the data into train and test sets stratified by target"""

        train_set, test_set = train_test_split(
            self.df, test_size=test_size, random_state=random_state, stratify=self.df[self.config.target]
        )

        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks catalog"""

        databricks_catalog = f"{self.config.catalog_name}.{self.config.schema_name}"
        train_set_name, test_set_name = self.config.train_set_name, self.config.test_set_name

        train_set_with_ts = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_ts = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        logger.info("Saving train set to Databricks catalog")
        train_set_with_ts.write.mode("append").saveAsTable(f"{databricks_catalog}.{train_set_name}")

        logger.info("Saving test set to Databricks catalog")
        test_set_with_ts.write.mode("append").saveAsTable(f"{databricks_catalog}.{test_set_name}")

        spark.sql(
            f"ALTER TABLE {databricks_catalog}.{train_set_name} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {databricks_catalog}.{test_set_name} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        logger.info(
            f"Train/Test sets saved to location: {databricks_catalog} under {train_set_name} and {test_set_name}"
        )
