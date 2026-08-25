from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv('VID2KB_DATA_DIR', 'data')
    deepseek_api_key: str = os.getenv('DEEPSEEK_API_KEY', '')
    deepseek_base_url: str = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    deepseek_model: str = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    dashscope_api_key: str = os.getenv('DASHSCOPE_API_KEY', '')
    dashscope_base_url: str = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    vision_model: str = os.getenv('VISION_MODEL', 'qwen-vl-max')
    embed_model: str = os.getenv('EMBED_MODEL', 'qwen3-embedding:0.6b')
    embed_dims: int = int(os.getenv('EMBED_DIMS', '1024'))
    ollama_base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    vector_store: str = os.getenv('VECTOR_STORE', 'pgvector')
    qdrant_url: str = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333')
    qdrant_collection: str = os.getenv('QDRANT_COLLECTION', 'vid2kb_docs')
    pgvector_database_url: str = os.getenv('PGVECTOR_DATABASE_URL', '')
    asr_backend: str = os.getenv('ASR_BACKEND', 'funasr')
    max_frames: int = int(os.getenv('MAX_FRAMES', '30'))
    frame_interval_seconds: int = int(os.getenv('FRAME_INTERVAL_SECONDS', '3'))
    run_driver: str = os.getenv('RUN_DRIVER', 'background')
    temporal_address: str = os.getenv('TEMPORAL_ADDRESS', '127.0.0.1:7233')
    artifact_store: str = os.getenv('ARTIFACT_STORE', 'fs')
    s3_endpoint_url: str = os.getenv('S3_ENDPOINT_URL', 'http://127.0.0.1:9000')
    s3_access_key: str = os.getenv('S3_ACCESS_KEY', 'minioadmin')
    s3_secret_key: str = os.getenv('S3_SECRET_KEY', 'minioadmin')
    s3_bucket: str = os.getenv('S3_BUCKET', 'vid2kb')
    s3_region: str = os.getenv('S3_REGION', 'us-east-1')

settings = Settings()
