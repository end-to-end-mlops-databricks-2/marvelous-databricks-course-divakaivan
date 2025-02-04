from typing import Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Project configuration"""

    raw_data_schema: Dict[str, str]
    catalog_name: str
    schema_name: str
    train_set_name: str
    test_set_name: str
    num_features: List[str]
    cat_features: List[str]
    target: str

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from yaml file"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
