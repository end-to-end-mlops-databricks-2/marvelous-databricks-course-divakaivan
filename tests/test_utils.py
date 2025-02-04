import pytest
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, 
    DoubleType, FloatType, BooleanType, DateType, TimestampType
)
from src.bank_marketing.utils import create_schema

def test_create_schema():
    field_dict = {
        "int_col": "IntegerType",
        "str_col": "StringType",
        "double_col": "DoubleType",
        "float_col": "FloatType",
        "bool_col": "BooleanType",
        "date_col": "DateType",
        "ts_col": "TimestampType"
    }
    
    expected_schema = StructType([
        StructField("int_col", IntegerType(), True),
        StructField("str_col", StringType(), True),
        StructField("double_col", DoubleType(), True),
        StructField("float_col", FloatType(), True),
        StructField("bool_col", BooleanType(), True),
        StructField("date_col", DateType(), True),
        StructField("ts_col", TimestampType(), True)
    ])
    
    assert create_schema(field_dict) == expected_schema

def test_create_schema_invalid_type():
    with pytest.raises(ValueError, match="Unsupported field type: UnknownType"):
        create_schema({"unknown": "UnknownType"})
