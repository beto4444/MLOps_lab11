from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.scripts.settings import Settings
from tokenizers import Tokenizer
import numpy as np
import onnxruntime as ort

from mangum import Mangum
settings = Settings()

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
app = FastAPI()


tokenizer = Tokenizer.from_file(str(settings.onnx_tokenizer_path))
emb_session = ort.InferenceSession(str(settings.onnx_embedding_model_path))
clf_session = ort.InferenceSession(str(settings.onnx_classifier_path))

class TextInput(BaseModel):
    text: str = Field(..., min_length=1)


class PredictionOutput(BaseModel):
    prediction: str


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API ver ONNX!"}


@app.get("/health")
def health_check():
    return {"status": "Alles klar Herr Kommissar!"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    # tokenize input
    encoded = tokenizer.encode(input_data.text)

    # prepare numpy arrays for ONNX
    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    # run embedding inference
    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    embeddings = emb_session.run(None, embedding_inputs)[0]
    # run classifier inference
    classifier_input_name = clf_session.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    prediction = clf_session.run(None, classifier_inputs)[0]

    label = SENTIMENT_MAP.get(int(prediction[0]), "unknown") # return this label as response
    return PredictionOutput(prediction=label)

handler = Mangum(app)