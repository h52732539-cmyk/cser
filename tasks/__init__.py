"""Tasks package — pluggable task adapters."""
from .base import BaseTask
from .retrieval_task import RetrievalTask
from .highlight_task import HighlightTask
from .face_task import FaceDetectionTask, FaceEmbeddingTask
from .scene_task import SceneClassificationTask
from .mock_models import (
    MockCLIPModel, MockHighlightModel, MockFaceDetector,
    MockFaceEmbedder, MockSceneClassifier, make_query_embeddings,
)
from . import real_models

__all__ = [
    "BaseTask",
    "RetrievalTask",
    "HighlightTask",
    "FaceDetectionTask",
    "FaceEmbeddingTask",
    "SceneClassificationTask",
    "MockCLIPModel",
    "MockHighlightModel",
    "MockFaceDetector",
    "MockFaceEmbedder",
    "MockSceneClassifier",
    "make_query_embeddings",
    "real_models",
]
