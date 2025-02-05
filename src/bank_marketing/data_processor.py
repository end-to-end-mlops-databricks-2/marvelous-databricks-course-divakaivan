from typing import Dict, Optional, Union

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from bank_marketing.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df
        self.config = config
        self._column_mapping = {
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
        }

    def clean_column_names(self, column_mapping: Optional[Dict[str, str]]):
        """Clean column names and rename them according to the mapping"""

        self.df.rename(
            columns=self._column_mapping if not column_mapping else column_mapping,
            inplace=True,
        )

    def preprocess(self):
        """Preprocess numerical, categorical features and target"""

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

    def split_data(
        self, test_size: Optional[float] = 0.2, random_state: Optional[int] = 42
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        """Split the data into train and test sets stratified by target"""

        train_set, test_set = train_test_split(
            self.df, test_size=test_size, random_state=random_state, stratify=self.df[self.config.target]
        )

        return train_set, test_set
