import pytest
import yaml
from pydantic import ValidationError

from src.bank_marketing.config import ProjectConfig


@pytest.fixture
def config_data():
    """Fixture to provide sample YAML config"""
    config_yaml = """
    data_path: "/data/path"
    raw_data_schema:
      col1: "IntegerType"
      col2: "StringType"
      col3: "StringType"
    catalog_name: "catalog"
    schema_name: "schema"
    train_set_name: "train_data"
    test_set_name: "test_data"
    num_features:
      - "col1"
    cat_features:
      - "col2"
    target: "col3"
    """
    config_dict = yaml.safe_load(config_yaml)

    return config_dict, ProjectConfig(**config_dict)


def test_create_project_config(config_data):
    """Test creating ProjectConfig object and comparing with expected"""
    config_dict, expected_config = config_data
    config = ProjectConfig(**config_dict)

    assert config == expected_config


def test_missing_field(config_data):
    """Test missing required field raises ValidationError"""
    config_dict, _ = config_data
    incomplete_config = config_dict.copy()
    del incomplete_config["target"]

    with pytest.raises(ValidationError):
        ProjectConfig(**incomplete_config)


def test_invalid_type(config_data):
    """Test invalid field type raises ValidationError"""
    config_dict, _ = config_data
    invalid_type_config = config_dict.copy()
    invalid_type_config["target"] = 123

    with pytest.raises(ValidationError):
        ProjectConfig(**invalid_type_config)


def test_from_yaml(tmp_path, config_data):
    """Test loading config from YAML"""
    config_dict, expected_config = config_data
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml.dump(config_dict))

    config = ProjectConfig.from_yaml(str(yaml_file))

    assert config == expected_config
