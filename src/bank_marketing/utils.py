from typing import Dict

from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def create_schema(field_dict: Dict[str, str]) -> StructType:
    """Create a schema for a PySpark DataFrame from a dictionary

    Args:
        field_dict (Dict[str, str]): A dictionary with field names as keys and field types as values

    Returns:
        StructType: PySpark schema
    """
    fields = []
    for field_name, field_type in field_dict.items():
        if field_type == "IntegerType":
            fields.append(StructField(field_name, IntegerType(), True))
        elif field_type == "StringType":
            fields.append(StructField(field_name, StringType(), True))
        elif field_type == "DoubleType":
            fields.append(StructField(field_name, DoubleType(), True))
        elif field_type == "FloatType":
            fields.append(StructField(field_name, FloatType(), True))
        elif field_type == "BooleanType":
            fields.append(StructField(field_name, BooleanType(), True))
        elif field_type == "DateType":
            fields.append(StructField(field_name, DateType(), True))
        elif field_type == "TimestampType":
            fields.append(StructField(field_name, TimestampType(), True))
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

    return StructType(fields)
