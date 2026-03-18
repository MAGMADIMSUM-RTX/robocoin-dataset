import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from robocoin_dataset.utils.le_path import (
    get_episodes_stats_jsonl_file,
    get_meta_info_file,
    get_parquet_files,
)


class DataPostProcessorBase:
    def __init__(
        self,
        convert_path: str | Path,
        data_post_process_type: str,
        data_feature_keys: set[str],
    ) -> None:
        self.convert_path: Path = Path(convert_path)
        if not self.convert_path.exists():
            raise FileNotFoundError(f"{self.convert_path} does not exist")

        if not (self.convert_path / "meta/info.json").exists():
            raise FileNotFoundError(f"{convert_path}/meta/info.json does not exist")

        self.parquet_files = get_parquet_files(self.convert_path)
        self.new_parquet_files = get_parquet_files(self.convert_path, data_post_process_type)

        self.data_features = data_feature_keys
        self.new_info_file = get_meta_info_file(self.convert_path, data_post_process_type)
        self.new_episodes_stats_file_path = get_episodes_stats_jsonl_file(
            self.convert_path, data_post_process_type
        )
        self.info_file_path = get_meta_info_file(self.convert_path)
        self._ep_idx: int | None = None

    # 将处理episode数据的准备工作放在这里
    def prepare_processing(self) -> None:
        pass

    # 该方法将ori_data进行后处理，返回结果为后处理后的数据
    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return ori_data.copy()

    # 该方法返回处理后的state数据名称
    def get_modified_feature_names(self) -> dict[str, list[str]]:
        return {}

    @property
    def episode_idx(self) -> int | None:
        return self._ep_idx

    def get_ori_episode_data(self, episode_idx: int) -> dict[str, np.ndarray | None]:
        if episode_idx >= len(self.parquet_files):
            raise ValueError(f"episode_idx {episode_idx} out of range")

        results = {}

        df = pd.read_parquet(self.parquet_files[episode_idx])

        for feature in self.data_features:
            if feature not in df:
                results[feature] = None
            else:
                results[feature] = np.array(df[feature].tolist())

        return results

    def _get_pa_type(self, np_dtype: np.dtype) -> pa.lib.DataType:
        mapping = {
            np.int32: pa.int32(),
            np.int64: pa.int32(),
            np.float32: pa.float32(),
            np.float64: pa.float32(),
            np.bool_: pa.bool_(),
        }
        return mapping.get(np.dtype(np_dtype).type, pa.from_numpy_dtype(np_dtype))

    def write_new_episode_file(self, new_data: dict[str, np.ndarray], episode_idx: int) -> None:
        if episode_idx >= len(self.new_parquet_files):
            raise ValueError(f"episode_idx {episode_idx} out of range")

        file_path = self.new_parquet_files[episode_idx]

        lengths = {key: arr.shape[0] for key, arr in new_data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Array length mismatch: {lengths}")

        try:
            arrays = []
            fields = []

            for col_name, arr in new_data.items():
                if arr.ndim == 1:
                    pa_type = self._get_pa_type(arr.dtype)
                    pa_array = pa.array(arr, type=pa_type)
                    arrays.append(pa_array)
                    fields.append(pa.field(col_name, pa_type))
                elif arr.ndim == 2:
                    # 使用 ListArray: 每个元素是一个 list
                    value_type = self._get_pa_type(arr.dtype)
                    list_type = pa.list_(value_type)
                    # 转换为 ListArray
                    pa_array = pa.array([row.tolist() for row in arr], type=list_type)
                    arrays.append(pa_array)
                    fields.append(pa.field(col_name, list_type))
                else:
                    raise ValueError(f"Unsupported array dimension: {arr.ndim} for '{col_name}'")

            schema = pa.schema(fields)
            table = pa.Table.from_arrays(arrays, schema=schema)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, file_path)

        except Exception as e:
            raise OSError(f"Failed to write episode {episode_idx} to {file_path}: {e}")

    def write_new_info_file(self) -> None:
        json_dict = {}
        json_dict["features"] = {}
        for feature_key, names in self.get_modified_feature_names().items():
            if names is not None:
                # 确保 names 是一个列表
                if not isinstance(names, list):
                    raise ValueError(
                        f"Feature names must be a list, got {type(names)} for '{feature_key}'"
                    )

                # 扁平化处理：如果列表中有嵌套列表，展开它们
                flattened_names = []
                for item in names:
                    if isinstance(item, list):
                        # 如果是列表，展开它
                        flattened_names.extend(item)
                    elif isinstance(item, str):
                        # 如果是字符串，直接添加
                        flattened_names.append(item)
                    else:
                        raise ValueError(
                            f"Feature name in '{feature_key}' must be a string or list, "
                            f"got {type(item)}: {item}"
                        )

                # 使用扁平化后的列表
                names = flattened_names

                # 检查是否有重复的名称
                if len(names) != len(set(names)):
                    duplicates = [name for name in names if names.count(name) > 1]
                    raise ValueError(
                        f"Feature '{feature_key}' contains duplicated names: {set(duplicates)}"
                    )

            json_dict["features"][feature_key] = {}
            json_dict["features"][feature_key]["names"] = names

        with open(self.new_info_file, "w") as f:
            json.dump(json_dict, f)

    def process(self) -> None:
        self.write_new_info_file()
        self.prepare_processing()
        self.episodes_stats = []
        for episode_idx in tqdm(
            range(len(self.parquet_files)), desc="Processing episodes", unit="episode"
        ):
            ori_data = self.get_ori_episode_data(episode_idx)
            self._ep_idx = episode_idx
            new_datas: dict[str, np.ndarray] = self.process_episode_data(ori_data)

            if set(new_datas.keys()) != set(self.data_features):
                print(f"new_datas keys {new_datas.keys()} != data_features {self.data_features}")
                print(
                    f"set(new_datas.keys())-set(self.data_features): {set(new_datas.keys()) - set(self.data_features)}"
                )
                print(
                    f"set(self.data_features)-set(new_datas.keys()): {set(self.data_features) - set(new_datas.keys())}"
                )
                raise ValueError(
                    f"new_datas keys {new_datas.keys()} != self.data_features {self.data_features}"
                )

            self.write_new_episode_file(new_datas, episode_idx)
            self.episodes_stats.append(self._compute_episode_stat(new_datas))
        self._write_new_episodes_stats_file()
        self._ep_idx = None

    def _compute_episode_stat(
        self, episode_data: dict[str, np.ndarray]
    ) -> dict[str, dict[str, list[float]]]:
        ep_stats = {}
        ep_stats["episode_index"] = self.episode_idx
        ep_stats["stats"] = {}
        for feature_key, data in episode_data.items():
            if data is None:
                continue
            ep_stats["stats"][feature_key] = {}
            ep_stats["stats"][feature_key]["mean"] = np.mean(data, axis=0).tolist()
            ep_stats["stats"][feature_key]["std"] = np.std(data, axis=0).tolist()
            ep_stats["stats"][feature_key]["min"] = np.min(data, axis=0).tolist()
            ep_stats["stats"][feature_key]["max"] = np.max(data, axis=0).tolist()
            ep_stats["stats"][feature_key]["count"] = [data.shape[0]]
        return ep_stats

    def _write_new_episodes_stats_file(self) -> None:
        with open(self.new_episodes_stats_file_path, "w") as f:
            for stat in self.episodes_stats:
                json.dump(stat, f)
                f.write("\n")
