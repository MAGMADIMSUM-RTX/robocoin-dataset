def tune_camera(config_path, robot_version):
    """
    专门用于调整相机的工具函数。
    1. 加载模型
    2. 启动 Viewer
    3. 允许用户用鼠标调整视角
    4. 每秒打印当前的 LookAt, Distance, Azimuth, Elevation
    """
    import os
    import time

    import mujoco
    import mujoco.viewer
    import yaml

    print(f"正在读取配置: {config_path} ...")
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    sim_cfg = full_config["sim"]
    xml_path = sim_cfg["xml_path"]

    if not os.path.exists(xml_path):
        # 尝试相对于 config 的路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(project_root, xml_path)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到 XML 文件: {xml_path}")

    print("加载 MuJoCo 模型...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("-" * 50)
    print("【操作说明】")
    print("1. 鼠标左键拖拽: 旋转视角")
    print("2. 鼠标右键拖拽: 平移视角 (改变 LookAt)")
    print("3. 鼠标滚轮: 缩放 (改变 Distance)")
    print("4. 找到满意角度后，直接复制下方打印的代码！")
    print("-" * 50)

    # 启动 Passive Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = 0
        while viewer.is_running():
            # 同步物理和渲染
            mujoco.mj_step(model, data)
            viewer.sync()

            now = time.time()
            if now - last_print > 1.0:
                # 获取当前 Viewer 的相机状态
                cam = viewer.cam

                print("\n>>> [当前相机参数]:")
                print(f"    lookat_x: {cam.lookat[0]:.3f}")
                print(f"    lookat_z: {cam.lookat[2]:.3f}")
                print(f"    lookat_y: {cam.lookat[1]:.3f}")
                print(f"    distance: {cam.distance:.3f}")
                print(f"    azimuth: {cam.azimuth:.3f}")
                print(f"    elevation: {cam.elevation:.3f}")

                last_print = now

            time.sleep(0.03)

if __name__ == "__main__":
    # 使用示例：直接在这里填入你的配置路径
    # 注意：不需要 repo_path，因为只看模型
    try:
        tune_camera(
            config_path="configs/sim_replay/agilex.yml",
            robot_version="agilex_cobot_magic_remote",  # 替换为你 yaml 里的真实 key
        )
    except Exception as e:
        print(f"发生错误: {e}")
