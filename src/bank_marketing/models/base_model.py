import mlflow
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from bank_marketing.config import ProjectConfig, Tags

"""
infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
"""

"""
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
model_prameters → Hyperparameters for the chosen model.
catalog_name, schema_name → Database schema names for Databricks tables.
"""


class BaseModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """ Initialize the model with project configuration """

        self.config = config
        self.spark = spark

        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.model_parameters = self.config.model_parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_base
        self.tags = tags.dict()

    def load_data(self):
        """
        Load training and testing data from Delta tables.

        Splits data into:
            Features (X_train, X_test)
            Target (y_train, y_test)
        """

        logger.info("Loading data from Databricks")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.{self.config.train_set_name}")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.{self.config.test_set_name}").toPandas()
        self.data_version = "0" #describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("Data successfully loaded.")

    def prepare_pipeline(self):
        """ Prepare scikit-learn pipeline for training """

        logger.info("Preparing pipeline")

        self.pipeline = Pipeline(steps=[
            ('hgbc_model', HistGradientBoostingClassifier(**self.model_parameters or {}))
        ])
        logger.info("Pipeline successfully prepared.")

    def train(self):
        """ Train the model """

        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self):
        """ Log the model """

        logger.info("Starting MLFlow experiment")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            report = classification_report(self.y_test, y_pred, output_dict=True)
            for key, value in report.items():
                if isinstance(value, dict):
                    for metric, score in value.items():
                        mlflow.log_metric(f"{key}_{metric}", score)
                else:
                    mlflow.log_metric(key, value)

            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.{self.config.train_set_name}",
                version=self.data_version
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="hgbc-pipeline-model",
                signature=signature
            )

    def register_model(self):
        """ Register model in Unity Catalog """

        logger.info("Registering model")
        registered_model = mlflow.register_model(
            model_uri=f'runs:/{self.run_id}/hgbc-pipeline-model',
            name=f"{self.catalog_name}.{self.schema_name}.bank_marketing_base_model",
            tags=self.tags
        )
        logger.info(f"Model registered under version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.bank_marketing_base_model",
            alias="latest-model",
            version=latest_version
        )

    def retrieve_current_run_dataset(self):
        """ Retrieve MLflow run dataset """

        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("Dataset source loaded.")

        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")

        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow and make predictions

        Args:
            input_data (pd.DataFrame): Pandas DataFrame containing input features for prediction

        Returns:
            pd.DataFrame: Pandas DataFrame with predictions
        """
        
        logger.info("Loading model from MLflow")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.bank_marketing_base_model@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("Model successfully loaded.")

        predictions = model.predict(input_data)

        return predictions
