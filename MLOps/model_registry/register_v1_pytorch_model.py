from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from mlflow import MlflowClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/inceptionresnetv1_imsfd_partial_finetuned.pth')
    parser.add_argument('--registered-name', default='TA-v2-InceptionResnetV1-FaceEmbedding')
    parser.add_argument('--model-version-tag', default='imsfd_partial_ft_v1')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    from facenet_pytorch import InceptionResnetV1
    try:
        payload = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location='cpu')
    state = payload.get('model_state') if isinstance(payload, dict) and 'model_state' in payload else payload
    model = InceptionResnetV1(pretrained=None, classify=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('TA-v2-Face-Recognition')
    example = np.zeros((1, 3, 160, 160), dtype=np.float32)
    with torch.inference_mode():
        output = model(torch.from_numpy(example)).detach().cpu().numpy()
    signature = infer_signature(example, output)

    with mlflow.start_run(run_name=f'register_{args.model_version_tag}') as run:
        mlflow.set_tags({
            'model_version': args.model_version_tag,
            'preprocessing_version': os.getenv('FACE_PREPROCESSING_VERSION', 'not_set'),
            'registration_type': 'imported_existing_model',
            'source_artifact': str(model_path.name),
        })
        info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name='face_embedding_backbone',
            registered_model_name=args.registered_name,
            signature=signature,
            input_example=example,
            pip_requirements=['torch', 'facenet-pytorch>=2.6.0', 'numpy'],
        )
    client = MlflowClient(tracking_uri=tracking_uri)
    version = str(info.registered_model_version) if info.registered_model_version else None
    if version:
        client.set_model_version_tag(args.registered_name, version, 'model_version', args.model_version_tag)
        client.set_registered_model_alias(args.registered_name, 'candidate', version)
    print('Registered model:', args.registered_name)
    print('Version:', version)
    print('Run ID:', run.info.run_id)
    print('Alias candidate assigned if supported by server.')
    print('Do not set production alias until validation and PAD checks are approved.')

if __name__ == '__main__':
    main()
