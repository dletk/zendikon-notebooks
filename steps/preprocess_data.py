import numpy as np
import pandas as pd
from zendikon.aml.tabular_data_step_decorator import TabularDataAmlPythonStepCompatible
from sklearn.preprocessing import MinMaxScaler


@TabularDataAmlPythonStepCompatible()
def preprocess_adult(features, cli_args=None, run=None):

    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'
    ]
    numerical_columns = list(set(features.columns) - set(categorical_columns))

    # convert categorical features to long
    features[categorical_columns] = features[categorical_columns].fillna(-1).astype("category")
    for c in categorical_columns:
        features[c] = features[c].cat.codes
        features[c] = features[c].astype(np.long)

    # create one-hot encodings
    features = pd.get_dummies(features, columns=categorical_columns)

    # scale numerical features
    scaler = MinMaxScaler()
    scaler.fit(features[numerical_columns])
    features[numerical_columns] = scaler.transform(features[numerical_columns])

    return [features]


if __name__ == "__main__":
    preprocess_adult()
