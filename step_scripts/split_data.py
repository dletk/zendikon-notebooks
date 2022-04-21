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
            name="split_date",
            helper_message="Date to split the dataset into train and validation sets",
            data_type=str,
            default_value="2012-01-01",
        )
    ],
)

def split_data(data, cli_args=None, run=None) -> List[pd.DataFrame]:
    train_data = data[data[cli_args.time_column_name] < cli_args.split_date]
    valid_data = data[data[cli_args.time_column_name] >= cli_args.split_date]
    return [train_data, valid_data]

if __name__ == "__main__":
    split_data()


