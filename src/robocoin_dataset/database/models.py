from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# =====================
# 枚举类型
# =====================
class TaskStatus(str, PyEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =====================
# 主表：DatasetDB
# =====================
class DatasetDB(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(255), unique=False, index=True, nullable=False)
    dataset_uuid = Column(String(255), unique=True, index=True, nullable=False)

    # 入库相关字段
    device_model = Column(String(100), nullable=True)
    device_model_version = Column(String(100), nullable=True)
    end_effector_type = Column(String(100), nullable=True)
    operation_platform_height = Column(Float, nullable=True)
    yaml_file_path = Column(String(255), nullable=True)
    data_path = Column(String(255), nullable=True)
    convert_path = Column(String(255), nullable=True)

    # 格式转换相关
    convert_test_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    convert_test_version = Column(Integer, nullable=True, default=0)
    convert_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    convert_version_ps = Column(Integer, nullable=True, default=0)
    convert_version = Column(Integer, nullable=True, default=0)
    total_episodes = Column(Integer, nullable=True, default=0)
    converted_episodes = Column(Integer, nullable=True, default=0)
    skipped_episodes = Column(Integer, nullable=True, default=0)
    convert_err_msg = Column(Text, nullable=True)

    # state 和 action后处理及Replay相关
    sa_dpp_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    sa_dpp_version_ps = Column(Integer, nullable=True, default=0)
    sa_dpp_version = Column(Integer, nullable=True, default=0)
    sa_dpp_err_msg = Column(Text, nullable=True)

    sim_replay_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    sim_replay_version_ps = Column(Integer, nullable=True, default=0)
    sim_replay_version = Column(Integer, nullable=True, default=0)
    sim_replay_error_msg = Column(Text, nullable=True)

    # 运动标注相关
    motion_annotation_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    motion_annotation_version_ps = Column(Integer, nullable=True, default=0)
    motion_annotation_version = Column(Integer, nullable=True, default=0)
    motion_annotation_err_msg = Column(Text, nullable=True)

    # subtask标注相关
    video_hash_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    video_hash_version_ps = Column(Integer, nullable=True, default=0)
    video_hash_version = Column(Integer, nullable=True, default=0)
    video_hash_err_msg = Column(Text, nullable=True)

    video_match_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    video_match_version_ps = Column(Integer, nullable=True, default=0)
    video_match_version = Column(Integer, nullable=True, default=0)
    video_match_err_msg = Column(Text, nullable=True)

    video_ori_subtask_annotation_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True
    )
    video_ori_subtask_annotation_version_ps = Column(Integer, nullable=True, default=0)
    video_ori_subtask_annotation_version = Column(Integer, nullable=True, default=0)
    video_ori_subtask_annotation_err_msg = Column(Text, nullable=True)

    video_opt_subtask_annotation_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True
    )
    video_opt_subtask_annotation_version_ps = Column(Integer, nullable=True, default=0)
    video_opt_subtask_annotation_version = Column(Integer, nullable=True, default=0)
    video_opt_subtask_annotation_err_msg = Column(Text, nullable=True)

    video_embed_subtask_annotation_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True
    )
    video_embed_subtask_annotation_version_ps = Column(Integer, nullable=True, default=0)
    video_embed_subtask_annotation_version = Column(Integer, nullable=True, default=0)
    video_embed_subtask_annotation_err_msg = Column(Text, nullable=True)

    scene_annotation_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    scene_annotation_version_ps = Column(Integer, nullable=True, default=0)
    scene_annotation_version = Column(Integer, nullable=True, default=0)
    scene_annotation_err_msg = Column(Text, nullable=True)

    # 数据合并相关
    data_merge_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    data_merge_version_ps_sta = Column(Integer, nullable=True, default=0)
    data_merge_version_ps_sa = Column(Integer, nullable=True, default=0)
    data_merge_version_ps_ma = Column(Integer, nullable=True, default=0)
    data_merge_version = Column(Integer, nullable=True, default=0)
    data_merge_err_msg = Column(Text, nullable=True)

    # Episode质量检测
    qc_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    qc_version_ps = Column(Integer, nullable=True, default=0)
    qc_version = Column(Integer, nullable=True, default=0)
    qc_err_msg = Column(Text, nullable=True)

    qced_repo_gen_path = Column(String(255), nullable=True)
    qced_repo_gen_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    qced_repo_gen_err_msg = Column(Text, nullable=True)
    qced_repo_gen_version = Column(Integer, nullable=True, default=0)
    qced_repo_gen_version_ps = Column(Integer, nullable=True, default=0)

    # dataLoader 检测相关
    data_loader_detection_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True
    )
    data_loader_detection_version_ps = Column(Integer, nullable=True, default=0)
    data_loader_detection_version = Column(Integer, nullable=True, default=0)
    data_loader_detection_err_msg = Column(Text, nullable=True)

    # 人工检测相关
    visualize_check_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    visualize_check_version_ps = Column(Integer, nullable=True, default=0)
    visualize_check_version = Column(Integer, nullable=True, default=0)
    visualize_check_err_msg = Column(Text, nullable=True)

    # 数据集上传相关
    ms_upload_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    ms_upload_version_ps = Column(Integer, nullable=True, default=0)
    ms_upload_version = Column(Integer, nullable=True, default=0)
    ms_upload_err_msg = Column(Text, nullable=True)

    huggingface_upload_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    huggingface_upload_version_ps = Column(Integer, nullable=True, default=0)
    huggingface_upload_version = Column(Integer, nullable=True, default=0)
    huggingface_upload_err_msg = Column(Text, nullable=True)

    # 数据集信息同步相关
    dataset_info_sync_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True)
    dataset_info_sync_version_ps_hf = Column(Integer, nullable=True, default=0)
    dataset_info_sync_version_ps_ms = Column(Integer, nullable=True, default=0)
    dataset_info_sync_version = Column(Integer, nullable=True, default=0)
    dataset_info_sync_err_msg = Column(Text, nullable=True)

    # 多对多关系（保持不变）
    scene_types = relationship(
        "SceneTypeDB", secondary="dataset_scene_types", back_populates="datasets"
    )
    task_descriptions = relationship(
        "TaskDescriptionDB", secondary="dataset_task_descriptions", back_populates="datasets"
    )
    objects = relationship("ObjectDB", secondary="dataset_objects", back_populates="datasets")
    atomic_actions = relationship(
        "AtomicActionDB", secondary="dataset_atomic_actions", back_populates="datasets"
    )


class SceneTypeDB(Base):
    __tablename__ = "scene_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    datasets = relationship(
        "DatasetDB", secondary="dataset_scene_types", back_populates="scene_types"
    )


class AtomicActionDB(Base):
    __tablename__ = "atomic_actions"
    id = Column(Integer, primary_key=True, index=True)
    action_name = Column(String(100), unique=True, nullable=False)
    datasets = relationship(
        "DatasetDB", secondary="dataset_atomic_actions", back_populates="atomic_actions"
    )


class TaskDescriptionDB(Base):
    __tablename__ = "task_descriptions"
    id = Column(Integer, primary_key=True, index=True)
    desc = Column(String(255), unique=True, index=True, nullable=False)
    datasets = relationship(
        "DatasetDB", secondary="dataset_task_descriptions", back_populates="task_descriptions"
    )


class ObjectDB(Base):
    __tablename__ = "object"
    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String(100), nullable=False, index=True)
    level1_category = Column(String(100), unique=False, nullable=True)
    level2_category = Column(String(100), unique=False, nullable=True)
    level3_category = Column(String(100), unique=False, nullable=True)
    level4_category = Column(String(100), unique=False, nullable=True)
    level5_category = Column(String(100), unique=False, nullable=True)
    datasets = relationship("DatasetDB", secondary="dataset_objects", back_populates="objects")


# =====================
# 多对多关联表（保持不变）
# =====================
dataset_scene_types = Table(
    "dataset_scene_types",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("scene_type_id", Integer, ForeignKey("scene_types.id"), primary_key=True),
)

dataset_task_descriptions = Table(
    "dataset_task_descriptions",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("task_description_id", Integer, ForeignKey("task_descriptions.id"), primary_key=True),
)

dataset_atomic_actions = Table(
    "dataset_atomic_actions",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("atomic_actions_id", Integer, ForeignKey("atomic_actions.id"), primary_key=True),
)

dataset_objects = Table(
    "dataset_objects",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("object_id", Integer, ForeignKey("object.id"), primary_key=True),
)


# 其他表结构保持不变
class VideoHashDB(Base):
    __tablename__ = "video_hash"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    ep_idx = Column(Integer, index=True, nullable=False)
    frame_num = Column(Integer, index=True, nullable=False)
    video_path = Column(String(255), index=False, nullable=False, unique=True)
    file_hash = Column(String(255), index=False, nullable=False)
    image_hashes = Column(Text, index=False, nullable=False)


class StAnnotationVideoDB(Base):
    __tablename__ = "st_annotation_video"

    id = Column(Integer, primary_key=True, index=True)
    video_url = Column(String(255), nullable=False, unique=True, index=True)
    local_video_path = Column(String(255), nullable=True, unique=True, index=True)
    download_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    video_hash_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    file_hash = Column(String(255), nullable=True, index=True)
    video_hash = Column(Text, nullable=True)
    frame_num = Column(Integer, nullable=True)

    annotations = relationship(
        "UrlVideoStAnnotationDB", back_populates="video", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_st_annotation_video_video_hash_status", "video_hash_status"),)


class UrlVideoStAnnotationDB(Base):
    __tablename__ = "url_video_st_annotation"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(
        Integer,
        ForeignKey("st_annotation_video.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    annotation = Column(Text, nullable=False)
    start_frame_idx = Column(Integer, nullable=False, index=True)
    end_frame_idx = Column(Integer, nullable=False, index=True)

    video = relationship("StAnnotationVideoDB", back_populates="annotations")

    __table_args__ = (
        Index(
            "ix_url_video_st_annotation_video_frames",
            "video_id",
            "start_frame_idx",
            "end_frame_idx",
        ),
    )


class VideoMatchDB(Base):
    __tablename__ = "video_match"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    url_video_id = Column(
        Integer,
        ForeignKey("st_annotation_video.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


class VideoStAnnotationDB(Base):
    __tablename__ = "video_st_annotation"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    start_frame_idx = Column(Integer, nullable=False, index=True)
    end_frame_idx = Column(Integer, nullable=False, index=True)
    annotation = Column(Text, nullable=False)


class VideoOptStAnnotationDB(Base):
    __tablename__ = "video_opt_st_annotation"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    start_frame_idx = Column(Integer, nullable=False, index=True)
    end_frame_idx = Column(Integer, nullable=False, index=True)
    annotation = Column(Text, nullable=False)


class DatasetHardLinkDB(Base):
    __tablename__ = "dataset_hard_link"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    hard_link_path = Column(String(255), index=True, nullable=True)


class EpisodeQcDB(Base):
    __tablename__ = "episode_qc"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    is_bad_episode = Column(Boolean, nullable=False)
    state_data_score = Column(Float, nullable=True)
    action_data_score = Column(Float, nullable=True)
    video_score = Column(Float, nullable=True)
    
    # ==================== Episode Data 算子单独得分 ====================
    episode_data_static_frame_rate_score = Column(Float, default=0.0)    # static_frame_rate 算子
    episode_data_static_joint_score = Column(Float, default=0.0)         # static_joint 算子
    
    # ==================== Episode Video 算子单独得分 ====================
    episode_video_max_frame_stable_then_jump_rate_score = Column(Float, default=0.0)  # max_frame_stable_then_jump_rate 算子
    episode_video_max_frame_jump_dist_score = Column(Float, default=0.0)              # max_frame_jump_dist 算子
    episode_video_color_shift_detection_score = Column(Float, default=0.0)           
    episode_video_consecutive_static_frames_score = Column(Float, default=0.0)
    episode_video_camera_resolution_consistency_score = Column(Float, default=0.0)