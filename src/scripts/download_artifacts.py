import boto3
import settings
import os

def download_artifacts(settings: settings.Settings):
    s3 = boto3.client("s3", region_name=settings.aws_region)

    os.makedirs(settings.local_model_dir, exist_ok=True)
    os.makedirs(settings.local_sentence_transformer_dir, exist_ok=True)

    s3.download_file(
        settings.s3_bucket_name,
        settings.s3_model_key,
        str(settings.local_classifier_path),
    )
    print("Classifier downloaded")

    model_dir = settings.s3_sentence_transformer_key
    resp = s3.list_objects_v2(
        Bucket=settings.s3_bucket_name,
        Prefix=model_dir
    )
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        relative_path = key[len(model_dir):]
        local_path = settings.local_sentence_transformer_dir / relative_path
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(
            settings.s3_bucket_name,
            key,
            str(local_path)
        )

    print("Sentence Transformer model downloaded")

if __name__ == "__main__":
    settings_instance = settings.Settings()
    download_artifacts(settings_instance)
