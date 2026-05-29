"""Core framework package."""
from .framework import LiteVTRFramework
from .framework_v2 import LiteVTRFrameworkV2, FrameworkV2Stats
from .subscription import TaskSubscription
from .types import Frame, Interval, InterestSignal, TaskResult, SamplingStage
from .prefilter import MetadataPrefilter, PrefilterResult
from .scheduler import UnifiedScheduler
from .two_stage import TwoStageController
from .cache import SharedFrameCache
from .frame_identity import FrameIdentity, byte_hash, phash, hamming
from .cross_task_cache import CrossTaskCache
from .adaptive_sampler import (
    UniformSampler, ContentFingerprintSampler, MVBasedSampler,
    QFrameSampler, QFrameConfig, HybridSampler,
)
from .offline_index import (
    OfflineIndex, OfflineIndexBuilder, VideoIndexEntry, build_protos,
)
from .query_planner import (
    QueryPlanner, QueryPlannerConfig, QueryPlan, QueryDifficulty,
)
from .metadata import (
    VideoMetadata, MOTION_CLASSES, GEO_CATEGORIES,
    extract_metadata, classify_motion_from_sensor, classify_geo,
    fill_derived_fields,
)
from .query_parser import QueryParser, QueryIntent
from .meta_filter import MetaFilter, FilterResult, fuse_scores, haversine_km
from .segment_aggregator import (
    Segment,
    SegmentAggregator,
    segments_mean_iou,
    boundary_mae,
)

__all__ = [
    "LiteVTRFramework",
    "LiteVTRFrameworkV2",
    "FrameworkV2Stats",
    "TaskSubscription",
    "Frame",
    "Interval",
    "InterestSignal",
    "TaskResult",
    "SamplingStage",
    "MetadataPrefilter",
    "PrefilterResult",
    "UnifiedScheduler",
    "TwoStageController",
    "SharedFrameCache",
    "FrameIdentity",
    "byte_hash",
    "phash",
    "hamming",
    "CrossTaskCache",
    "UniformSampler",
    "ContentFingerprintSampler",
    "MVBasedSampler",
    "QFrameSampler",
    "QFrameConfig",
    "HybridSampler",
    "OfflineIndex",
    "OfflineIndexBuilder",
    "VideoIndexEntry",
    "build_protos",
    "QueryPlanner",
    "QueryPlannerConfig",
    "QueryPlan",
    "QueryDifficulty",
    "VideoMetadata",
    "MOTION_CLASSES",
    "GEO_CATEGORIES",
    "extract_metadata",
    "classify_motion_from_sensor",
    "classify_geo",
    "fill_derived_fields",
    "QueryParser",
    "QueryIntent",
    "MetaFilter",
    "FilterResult",
    "fuse_scores",
    "haversine_km",
    "Segment",
    "SegmentAggregator",
    "segments_mean_iou",
    "boundary_mae",
]
