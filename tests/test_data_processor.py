import math

from src.bank_marketing.config import ProjectConfig
from src.bank_marketing.data_processor import DataProcessor

import pytest
import pandas as pd

@pytest.fixture
def mock_config():
    """Return a mock configuration and DataFrame"""
    config = ProjectConfig(
        raw_data_schema={'col1': 'IntegerType', 'col2': 'StringType', 'col3': 'StringType'},
        catalog_name='catalog_name',
        schema_name='schema_name',
        train_set_name='train_set',
        test_set_name='test_set',
        num_features=['age', 'balance'],
        cat_features=['marital'],
        target='has_subscribed'
    )
    df = pd.DataFrame({
        'age': [30, 20, 25, 54, 43, 26, 67, 71, 20, 25, 54, 43, 26, 67, 71],
        'balance': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], 
        'marital': ['married', 'single', 'married', 'single', 'married', 'single', 'married', 'single', 'single', 'married', 'single', 'married', 'single', 'married', 'single'],
        'has_subscribed': ['no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no']
    })

    return config, df

def test_data_processor_initialization(mock_config):
    """Test that the data processor is initialized correctly"""
    config, df = mock_config
  
    processor = DataProcessor(df, config)
    assert isinstance(processor.config, ProjectConfig) 
    assert isinstance(processor.df, pd.DataFrame)

def test_preprocess(mock_config):
    """Test that the data is preprocessed correctly"""
    config, df = mock_config

    processor = DataProcessor(df, config)
    processor.preprocess()
  
    assert pd.api.types.is_numeric_dtype(processor.df["age"])
    assert pd.api.types.is_numeric_dtype(processor.df["balance"])
    
    assert set(processor.df["marital"].unique()) == {0, 1}
    assert set(processor.df["has_subscribed"].unique()) == {0, 1}
    
    expected_columns = ['age', 'balance', 'marital', 'has_subscribed']
    assert all(col in processor.df.columns for col in expected_columns)

    needed_cols = config.num_features + config.cat_features + [config.target]
    assert list(processor.df.columns) == needed_cols

def test_split_data(mock_config):
    """Test that the data is split correctly"""
    config, df = mock_config
    processor = DataProcessor(df, config)

    train, test = processor.split_data()
    train_size = math.floor(df.shape[0] * 0.8)
    test_size = df.shape[0] - train_size
    
    assert len(train) == train_size
    assert len(test) == test_size

def test_stratification(mock_config):
    """Test that the stratification is working as expected"""
    config, df = mock_config
    processor = DataProcessor(df, config)
    train, test = processor.split_data(test_size=0.5)
    
    original_subscribed_ratio = (df["has_subscribed"] == "yes").mean()

    train_subscribed_ratio = (train["has_subscribed"] == "yes").mean()
    test_subscribed_ratio = (test["has_subscribed"] == "yes").mean()

    assert abs(original_subscribed_ratio - train_subscribed_ratio) <= 0.05
    assert abs(original_subscribed_ratio - test_subscribed_ratio) <= 0.05
