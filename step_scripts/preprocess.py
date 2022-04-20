#!/usr/bin/env python
# coding: utf-8

# # Preprocessing script for the beer dataset
import pandas as pd
from typing import List

from zendikon.aml.tabular_data_step_decorator import TabularDataAmlPythonStepCompatible
from zendikon.aml.types import StepArgument


@TabularDataAmlPythonStepCompatible(
    step_arguments=[
        StepArgument(
            name="time_column_name",
            helper_message="Time column name for training",
            data_type=str,
            default_value="DATE",
        ),
        StepArgument(
            name="target_column_name",
            helper_message="Target column name to predict",
            data_type=str,
            default_value="BeerProduction",
        ),
    ],
)
def preprocess_data(data, cli_args=None, run=None) -> pd.DataFrame:
    cols = [cli_args.time_column_name, cli_args.target_column_name]
    result = data.loc[:, cols]

    return result

if __name__ == "__main__":
    preprocess_data()





