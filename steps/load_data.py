import numpy as np
import pandas as pd
from zendikon.aml.tabular_data_step_decorator import TabularDataAmlPythonStepCompatible
from sklearn.datasets import fetch_openml

from zendikon.aml.types import StepArgument


@TabularDataAmlPythonStepCompatible(step_arguments=[StepArgument("active", "Dummy argument", bool, False, required=True),
                                                    StepArgument("ratio", "a float", float, required=True)])
def load_adult(cli_args=None, run=None):
    adult = fetch_openml(name='adult', version=2)

    if cli_args.active:
        print("Active is True")
    else:
        print("Active is False")

    print("Type of active: ", type(cli_args.active))
    print("Value for ratio: ", cli_args.ratio)

    features = pd.DataFrame(adult.data, columns=adult.feature_names)

    targets = np.zeros_like(adult.target, dtype=np.long)
    targets[adult.target == '>50K'] = 1
    targets = pd.DataFrame(targets)

    return [features, targets]


if __name__ == "__main__":
    load_adult()
