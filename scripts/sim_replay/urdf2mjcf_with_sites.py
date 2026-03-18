import traceback
from argparse import ArgumentParser
from pathlib import Path

# add_site_to_body.py
from xml.etree import ElementTree as ET

import mujoco


def add_site_to_body_in_mjcf(
    mjcf_path: Path | str,
    output_path: Path | str,
    left_eef_body_name: str | None = None,
    right_eef_body_name: str | None = None,
    site_sphere_size: float = 0.05,
) -> None:
    """
    在指定 body 中添加 site，并保存为新 MJCF 文件

    Args:
        mjcf_path: str or Path, 原始 MJCF 文件路径
        body_name: str, 要添加 site 的 body 名称
        site_config: dict, site 参数，如 {"name": "xxx", "pos": "0 0 0", ...}
        output_path: str or Path, 输出文件路径（默认为原文件加 _site）
    """
    mjcf_path = Path(mjcf_path)
    output_path = Path(output_path)

    left_eef_site_name = "left_eef_site"
    right_eef_site_name = "right_eef_site"

    # 1. 解析 MJCF XML
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    if left_eef_body_name is None and right_eef_body_name is None:
        raise ValueError("至少需要指定一个末端执行器 body 名称")

    # 2. 查找指定 body
    bodies = [body for body in root.iter("body")]
    bodies_dict = {body.get("name"): body for body in bodies}

    left_eef_body = bodies_dict.get(left_eef_body_name, None)
    right_eef_body = bodies_dict.get(right_eef_body_name, None)
    if left_eef_body is None:
        raise ValueError(f"未找到名为 {left_eef_body_name} 的 body")

    if right_eef_body is None:
        raise ValueError(f"未找到名为 {right_eef_body_name} 的 body")

    rgba = "0 1 0 0.5"  # 绿色
    # rgba = "1 0 0 0.5"  # 红色

    site_elem = ET.SubElement(left_eef_body, "site")
    site_elem.set("name", left_eef_site_name)
    site_elem.set("pos", "0 0 0")
    site_elem.set("size", str(site_sphere_size))
    site_elem.set("rgba", rgba)
    site_elem.set("type", "sphere")

    site_elem_x = ET.SubElement(left_eef_body, "site")
    site_elem_x.set("name", left_eef_site_name + "_xaxis")
    site_elem_x.set("pos", "0.025 0 0")
    site_elem_x.set("size", "0.025 0.0025 0.0025")
    site_elem_x.set("rgba", rgba)
    site_elem_x.set("type", "box")

    site_elem_y = ET.SubElement(left_eef_body, "site")
    site_elem_y.set("name", left_eef_site_name + "_yaxis")
    site_elem_y.set("pos", "0 0.05 0")
    site_elem_y.set("size", "0.0025 0.05 0.0025")
    site_elem_y.set("rgba", rgba)
    site_elem_y.set("type", "box")

    site_elem_z = ET.SubElement(left_eef_body, "site")
    site_elem_z.set("name", left_eef_site_name + "_zaxis")
    site_elem_z.set("pos", "0 0 0.1")
    site_elem_z.set("size", "0.0025 0.0025 0.1")
    site_elem_z.set("rgba", rgba)
    site_elem_z.set("type", "box")

    rgba = "1 0 0 0.5"  # 红色
    site_elem = ET.SubElement(right_eef_body, "site")
    site_elem.set("name", right_eef_site_name)
    site_elem.set("pos", "0 0 0")
    site_elem.set("size", str(site_sphere_size))
    site_elem.set("rgba", rgba)
    site_elem.set("type", "sphere")

    site_elem_x = ET.SubElement(right_eef_body, "site")
    site_elem_x.set("name", right_eef_site_name + "_xaxis")
    site_elem_x.set("pos", "0.025 0 0")
    site_elem_x.set("size", "0.025 0.0025 0.0025")
    site_elem_x.set("rgba", rgba)
    site_elem_x.set("type", "box")

    site_elem_y = ET.SubElement(right_eef_body, "site")
    site_elem_y.set("name", right_eef_site_name + "_yaxis")
    site_elem_y.set("pos", "0 0.05 0")
    site_elem_y.set("size", "0.0025 0.05 0.0025")
    site_elem_y.set("rgba", rgba)
    site_elem_y.set("type", "box")

    site_elem_z = ET.SubElement(right_eef_body, "site")
    site_elem_z.set("name", right_eef_site_name + "_zaxis")
    site_elem_z.set("pos", "0 0 0.1")
    site_elem_z.set("size", "0.0025 0.0025 0.1")
    site_elem_z.set("rgba", rgba)
    site_elem_z.set("type", "box")
    # 4. 保存修改后的 MJCF
    tree.write(output_path, encoding="utf-8", xml_declaration=True, method="xml")
    print(f"📁 Saved to: {output_path}")


# -------------------------------
# 1. 加载模型路径
# -------------------------------
argparser = ArgumentParser()
argparser.add_argument("urdf_file_path", type=str, help="Path to the URDF or MJCF model file.")
argparser.add_argument(
    "--left_eef_body_name", type=str, default=None, help="left end effector body name."
)

argparser.add_argument(
    "--right_eef_body_name", type=str, default=None, help="right end effector body name."
)
args = argparser.parse_args()

urdf_file_path = Path(args.urdf_file_path).expanduser().resolve()
left_eef_body_name = args.left_eef_body_name
right_eef_body_name = args.right_eef_body_name

try:
    
    model = mujoco.MjModel.from_xml_path(str(urdf_file_path))
    temp_mjcf_path = urdf_file_path.parent / (urdf_file_path.stem + "_temp.xml")

    mujoco.mj_saveLastXML(str(temp_mjcf_path), model)
    output_mjcf_path = urdf_file_path.parent / (urdf_file_path.stem + "_with_sites.xml")
    add_site_to_body_in_mjcf(
        mjcf_path=temp_mjcf_path,
        output_path=output_mjcf_path,
        left_eef_body_name=left_eef_body_name,
        right_eef_body_name=right_eef_body_name,
    )
    temp_mjcf_path.unlink()  # 删除临时文件


except Exception as e:
    print("❌ Conversion failed:")
    traceback.print_exc()
    exit(1)

"""usage:

"""
