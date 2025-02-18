import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from bank_marketing.models.base_model import BaseModel
from bank_marketing.config import ProjectConfig, Tags

@pytest.fixture
def mock_config():
    return ProjectConfig(
        raw_data_schema={"col1": "IntegerType", "col2": "StringType", "col3": "StringType"},
        catalog_name="catalog_name",
        schema_name="schema_name",
        train_set_name="train_set",
        test_set_name="test_set",
        num_features=["age", "balance"],
        cat_features=["marital"],
        target="has_subscribed",
        model_parameters={},
        experiment_name_base="/experiment/base_dir",
    )

@pytest.fixture
def mock_tags():
    return Tags(git_sha='test_sha', branch='test_branch')

@pytest.fixture
def mock_spark():
    spark = MagicMock()
    spark.table.return_value.toPandas.return_value = pd.DataFrame({
        'age': [25, 35, 45],
        'balance': [1000, 2000, 3000],
        'marital': [0, 1, 2],
        'has_subscribed': [0, 1, 0]
    })
    return spark

@pytest.fixture
def model(mock_config, mock_tags, mock_spark):
    return BaseModel(config=mock_config, tags=mock_tags, spark=mock_spark)

def test_load_data(model):
    model.load_data()
    assert not model.X_train.empty
    assert not model.X_test.empty
    assert not model.y_train.empty
    assert not model.y_test.empty

def test_load_latest_model_and_predict(model):
    test_input = pd.DataFrame({'age': [30], 'balance': [1500], 'marital': ['single']})
    with patch("mlflow.sklearn.load_model") as mock_load_model:
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_load_model.return_value = mock_model
        predictions = model.load_latest_model_and_predict(test_input)
        assert predictions[0] == 1
