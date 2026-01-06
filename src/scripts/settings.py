from pydantic_settings import BaseSettings
from pathlib import Path
class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "mlops11-wiadro-bs"

    s3_model_key: str = "classifier.joblib"
    s3_sentence_transformer_key: str = "sentence_transformer.model/"

    local_model_dir: Path = Path("model")
    local_sentence_transformer_dir: Path = local_model_dir / "sentence_transformer.model"
    local_classifier_path: Path = local_model_dir / "classifier.joblib"

    onnx_model_dir: Path = Path("onnx_model")
    onnx_embedding_model_path: Path = onnx_model_dir / "sentence_embeddings.onnx"
    onnx_classifier_path: Path = onnx_model_dir / "classifier.onnx"
    onnx_tokenizer_path: Path = onnx_model_dir / "tokenizer.json"

    embedding_dim: int = 384