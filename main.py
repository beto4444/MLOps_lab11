from src.scripts.settings import Settings
from src.scripts.export_sentence_transformer_to_onnx import export_model_to_onnx
from src.scripts.export_classifier_to_onnx import export_classifier_to_onnx
from src.scripts.download_artifacts import download_artifacts

if __name__ == "__main__":
    settings = Settings()
    download_artifacts(settings)
    export_model_to_onnx(settings)
    export_classifier_to_onnx(settings)