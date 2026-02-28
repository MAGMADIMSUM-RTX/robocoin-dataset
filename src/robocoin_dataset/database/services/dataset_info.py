import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    AtomicActionDB,
    DatasetDB,
    ObjectDB,
    SceneTypeDB,
    TaskDescriptionDB,
    dataset_atomic_actions,
    dataset_objects,
    dataset_scene_types,
    dataset_task_descriptions,
)

logger = logging.getLogger(__name__)

def _do_upsert(yaml_data: Dict[str, Any], session) -> None:
    """
    核心Upsert逻辑（复用session）
    :param yaml_data: YAML解析后的字典
    :param session: 数据库会话
    """
    # 1. 基础字段处理（包含 dataset_name_id）
    dataset_data = {
        "dataset_name": yaml_data["dataset_name"],  # 保留原始名称不修改
        "dataset_uuid": yaml_data["dataset_uuid"],  # 不自动生成，依赖外部传入
        "end_effector_type": yaml_data.get("end_effector_type"),
        "operation_platform_height": yaml_data.get("operation_platform_height"),
        "yaml_file_path": yaml_data.get("yaml_file_path"),
        "dataset_name_id": yaml_data.get("dataset_name_id", 0),  # 强制写入序号（默认0）
        "dataset_batch_number": yaml_data.get("dataset_batch_number", 0),
        "data_path": yaml_data.get("data_path"),  # 新增data_path字段
    }
    
    # 处理 device_model（列表转字符串）
    device_model = yaml_data.get("device_model")
    if isinstance(device_model, list):
        dataset_data["device_model"] = device_model[0] if device_model else None
    else:
        dataset_data["device_model"] = device_model
    
    # 过滤空值（但保留 dataset_name_id，即使为0）
    dataset_data = {
        k: v for k, v in dataset_data.items() 
        if v is not None or k == "dataset_name_id"  # 确保dataset_name_id始终存在
    }

    # 2. 多对多字段预处理
    # 任务描述
    task_descs = yaml_data.get("task_instruction", [])
    
    # 原子动作
    atomic_actions = yaml_data.get("atomic_actions", [])
    # 物品列表
    yaml_objects = yaml_data.get("objects", [])

    # 3. 场景类型处理（适配 scene_level1/scene_level2）
    scene_types: List[SceneTypeDB] = []
    # 处理一级场景
    scene_level1 = yaml_data.get("scene_level1")
    if scene_level1:
        st1 = session.query(SceneTypeDB).filter_by(name=scene_level1).first()
        if not st1:
            # 补充 level_id=1（匹配之前插入的场景数据）
            st1 = SceneTypeDB(name=scene_level1, level_id=1)
            session.add(st1)
        scene_types.append(st1)
    
    # 处理二级场景
    scene_level2 = yaml_data.get("scene_level2")
    if scene_level2:
        st2 = session.query(SceneTypeDB).filter_by(name=scene_level2).first()
        if not st2:
            # 补充 level_id=2（匹配之前插入的场景数据）
            st2 = SceneTypeDB(name=scene_level2, level_id=2)
            session.add(st2)
        scene_types.append(st2)

    # 4. 任务描述处理（保证全局唯一）
    task_desc_map = {}
    for desc in task_descs:
        if not desc or not isinstance(desc, str):
            continue
        td = session.query(TaskDescriptionDB).filter_by(desc=desc).first()
        if not td:
            td = TaskDescriptionDB(desc=desc)
            session.add(td)
            session.flush()  # 立即生成 id
        task_desc_map[td.id] = td
    task_descriptions = list(task_desc_map.values())

    # 5. 物品处理（仅处理 object_name，不保存颜色/分类字段）
    db_objects: List[ObjectDB] = []
    for obj in yaml_objects:
        if not isinstance(obj, dict):
            continue
        obj_name = obj.get("object_name")
        if not obj_name:
            continue
        
        # 查询或创建物品（按 object_name 全局唯一）
        ob = session.query(ObjectDB).filter_by(object_name=obj_name).first()
        if not ob:
            ob = ObjectDB(object_name=obj_name)
            session.add(ob)  # 仅创建/复用，不赋值颜色/分类
        db_objects.append(ob)

    # 6. Dataset 本体处理（UPSERT：存在则更新，不存在则创建）
    # 按 UUID 查找数据集（主键）
    ds = session.query(DatasetDB).filter_by(dataset_uuid=dataset_data["dataset_uuid"]).first()
    if not ds:
        # 新建数据集：完整写入所有字段（含 dataset_name_id）
        ds = DatasetDB(** dataset_data)
        session.add(ds)
        logger.info(f"创建新数据集: {dataset_data['dataset_name']} (ID: {dataset_data['dataset_name_id']}, UUID: {dataset_data['dataset_uuid']})")
    else:
        # 增量更新字段（重点：强制更新 dataset_name_id）
        for k, v in dataset_data.items():
            setattr(ds, k, v)
        logger.info(f"更新数据集: {dataset_data['dataset_name']} (ID: {dataset_data['dataset_name_id']}, UUID: {dataset_data['dataset_uuid']})")
    session.flush()  # 生成 ds.id，供后续多对多关联使用

    # 7. 建立多对多关联（检查是否已存在，避免重复）
    # 7.1 物品关联
    for obj in db_objects:
        exists = session.execute(
            select(dataset_objects.c.object_id)
            .where(dataset_objects.c.dataset_id == ds.id)
            .where(dataset_objects.c.object_id == obj.id)
        ).first() is not None
        
        if not exists:
            session.execute(
                dataset_objects.insert().values(dataset_id=ds.id, object_id=obj.id)
            )

    # 7.2 场景类型关联
    for st in scene_types:
        exists = session.execute(
            select(dataset_scene_types.c.scene_type_id)
            .where(dataset_scene_types.c.dataset_id == ds.id)
            .where(dataset_scene_types.c.scene_type_id == st.id)
        ).first() is not None
        
        if not exists:
            session.execute(
                dataset_scene_types.insert().values(dataset_id=ds.id, scene_type_id=st.id)
            )

    # 7.3 任务描述关联
    for td in task_descriptions:
        exists = session.execute(
            select(dataset_task_descriptions.c.task_description_id)
            .where(dataset_task_descriptions.c.dataset_id == ds.id)
            .where(dataset_task_descriptions.c.task_description_id == td.id)
        ).first() is not None
        
        if not exists:
            session.execute(
                dataset_task_descriptions.insert().values(
                    dataset_id=ds.id, task_description_id=td.id
                )
            )

    # 7.4 原子动作关联
    for action_name in atomic_actions:
        if not action_name or not isinstance(action_name, str):
            continue
        
        # 查询或创建原子动作
        atomic_action = session.query(AtomicActionDB).filter_by(action_name=action_name).first()
        if not atomic_action:
            atomic_action = AtomicActionDB(action_name=action_name)
            session.add(atomic_action)
            session.flush()  # 生成 id
        
        # 检查关联是否存在
        exists = session.execute(
            select(dataset_atomic_actions.c.atomic_actions_id)
            .where(dataset_atomic_actions.c.dataset_id == ds.id)
            .where(dataset_atomic_actions.c.atomic_actions_id == atomic_action.id)
        ).first() is not None
        
        if not exists:
            session.execute(
                dataset_atomic_actions.insert().values(
                    dataset_id=ds.id, atomic_actions_id=atomic_action.id
                )
            )

    logger.info(
        "Upsert完成 - 数据集: %s | 序号ID: %s | UUID: %s | 数据路径: %s", 
        dataset_data.get("dataset_name"),
        dataset_data.get("dataset_name_id"),
        dataset_data["dataset_uuid"],
        dataset_data.get("data_path")
    )

def upsert_dataset_info(yaml_data: Dict[str, Any], db_path: str = None, session: Optional[Any] = None) -> None:
    """
    将 YAML 字典写入数据库。
    :param yaml_data: YAML解析后的字典
    :param db_path: 数据库路径（session 为 None 时生效）
    :param session: 已有的数据库会话（优先使用）
    """
    # 如果传入了 session，直接使用；否则创建新的
    if session is not None:
        try:
            _do_upsert(yaml_data, session)
        except Exception as e:
            logger.error(f"Upsert失败（使用外部session）: {yaml_data.get('dataset_name')} - {e}")
            raise
        return
    
    # 原有逻辑（无session时）
    if db_path is None:
        raise ValueError("db_path 不能为空（当 session 为 None 时）")
    
    db = DatasetDatabase(db_path)
    with db.with_session() as inner_session:
        try:
            _do_upsert(yaml_data, inner_session)
            inner_session.commit()
        except Exception as e:
            inner_session.rollback()
            logger.error(
                "Upsert失败 - 数据集: %s | 序号ID: %s | 错误: %s",
                yaml_data.get("dataset_name"),
                yaml_data.get("dataset_name_id"),
                str(e),
                exc_info=True
            )
            raise