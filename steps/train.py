import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from azureml.core.model import Model

from zendikon.aml.tabular_data_step_decorator import TabularDataAmlPythonStepCompatible
from zendikon.aml.types import StepArgument


@TabularDataAmlPythonStepCompatible(step_arguments=[StepArgument("test_size", "The test size ratio for train test split", float, 0.2)])
def train_lr(features, targets, cli_args=None, run=None):

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=cli_args.test_size, random_state=42)

    model = LogisticRegression(solver='liblinear')

    model.fit(features_train, targets_train)

    targets_pred_train = model.predict(features_train)
    targets_pred_test = model.predict(features_test)

    # print(classification_report(targets_pred_train, targets_train))

    # print(classification_report(targets_pred_test, targets_test))

    if run is not None:
        run.log("f1_score", f1_score(targets_test, targets_pred_test))
        run.log("recall", recall_score(targets_test, targets_pred_test))
        run.log("precision", precision_score(targets_test, targets_pred_test))
        run.log("accuracy", accuracy_score(targets_test, targets_pred_test))

        initial_type = [('float_input', FloatTensorType([None, features_train.shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open("./outputs/lr_model.onnx", "wb") as f:
            f.write(onx.SerializeToString())

        run.upload_file("outputs/lr_model.onnx", "outputs/lr_model.onnx")
        run.register_model("simple_pipeline_LR", "./outputs/lr_model.onnx", model_framework=Model.Framework.ONNX)

    return [pd.DataFrame(targets_pred_test)]


if __name__ == "__main__":
    train_lr()
