# export_classifier_to_onnx.py

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from settings import Settings


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.local_classifier_path}...")
    classifier = joblib.load(settings.local_classifier_path)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(classifier, initial_types=initial_type)

    print(f"Saving ONNX model to {settings.onnx_classifier_path}...")
    with open(settings.onnx_classifier_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("ONNX model saved.")
    # TODO: save the onnx_model to settings.onnx_classifier_path

if __name__ == "__main__":
    settings = Settings()
    export_classifier_to_onnx(settings)
