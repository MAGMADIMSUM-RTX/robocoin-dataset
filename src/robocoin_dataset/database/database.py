from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import yaml

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from robocoin_dataset.database.models import Base


class DatasetDatabase:
    def __init__(self, db_file: str | Path = "postgresql_config.yaml") -> None:
        """
        初始化PostgreSQL数据库连接（从YAML配置文件读取参数）
        
        Args:
            config_path: YAML配置文件路径，默认为当前目录下的postgresql_config.yaml
        """
        self.config_path = Path(db_file)
        self.db_config = self._load_config()
        self.engine = None
        self.session_local = None
        self._initialize()
        self._create_tables()

    def _load_config(self) -> dict:
        """从YAML文件加载数据库配置"""
        try:
            # 检查配置文件是否存在
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在：{self.config_path.absolute()}")
            
            # 读取并解析YAML文件
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # 验证配置完整性
            pg_config = config.get("postgresql", {})
            required_keys = ["user", "password", "host", "database"]
            missing_keys = [key for key in required_keys if key not in pg_config]
            
            if missing_keys:
                raise ValueError(f"配置文件缺少必要参数：{', '.join(missing_keys)}")
            
            # 设置默认值
            pg_config.setdefault("port", 5432)
            pg_config.setdefault("pool_size", 10)
            pg_config.setdefault("max_overflow", 20)
            pg_config.setdefault("echo", False)
            
            return pg_config
        
        except yaml.YAMLError as e:
            raise RuntimeError(f"解析YAML配置文件失败：{e}")
        except Exception as e:
            raise RuntimeError(f"加载数据库配置失败：{e}")

    def _initialize(self) -> None:
        """初始化数据库引擎和session工厂"""
        # 构建PostgreSQL连接字符串
        db_url = (
            f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        
        # PostgreSQL引擎配置
        self.engine = create_engine(
            db_url,
            pool_size=self.db_config["pool_size"],
            max_overflow=self.db_config["max_overflow"],
            pool_pre_ping=True,
            echo=self.db_config["echo"]
        )
        
        self.session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库session生成器"""
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def with_session(self) -> Generator[Session, None, None]:
        """
        安全的上下文管理器，确保session正确关闭
        完全依赖PostgreSQL的事务和锁机制保证数据一致性
        """
        gen = self.get_session()
        session = next(gen)
        try:
            yield session
            session.commit()  # 显式提交事务
        except Exception as e:
            session.rollback()  # 异常时回滚
            raise e
        finally:
            gen.close()

    def _create_tables(self) -> None:
        """创建所有数据表（如果不存在）"""
        Base.metadata.create_all(bind=self.engine)


if __name__ == "__main__":
    # 方式1：使用默认配置文件（postgresql_config.yaml）
    try:
        db = DatasetDatabase("/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml")
        print("数据库实例创建成功！")
        
    
    except Exception as e:
        print(f"初始化失败：{e}")



# ALTER TABLE episode_qc
# ADD COLUMN IF NOT EXISTS episode_data_static_frame_rate_score FLOAT DEFAULT 0.0,
# ADD COLUMN IF NOT EXISTS episode_data_static_joint_score FLOAT DEFAULT 0.0,
# ADD COLUMN IF NOT EXISTS episode_video_max_frame_stable_then_jump_rate_score FLOAT DEFAULT 0.0,
# ADD COLUMN IF NOT EXISTS episode_video_max_frame_jump_dist_score FLOAT DEFAULT 0.0,
# ADD COLUMN IF NOT EXISTS episode_video_color_shift_detection_score FLOAT DEFAULT 0.0;

# -- 验证字段是否添加成功
# SELECT column_name 
# FROM information_schema.columns 
# WHERE table_name = 'episode_qc' 
# AND column_name IN (
#     'episode_data_static_frame_rate_score',
#     'episode_data_static_joint_score',
#     'episode_video_max_frame_stable_then_jump_rate_score',
#     'episode_video_max_frame_jump_dist_score',
#     'episode_video_color_shift_detection_score'
# );