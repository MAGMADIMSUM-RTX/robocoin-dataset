import numpy as np

# ====================== 改成你的 .npz 文件路径 ======================
file_path = "/media/liuyou/aaa/20技能/proc/CoRobot/Galaxea_R1_Lite_pick_and_place_Mango_1_1476/eef_full_ep000000.npz"

# 加载数据
data = np.load(file_path, allow_pickle=True)

print("="*60)
print("📄 文件名：", file_path)
print("🔑 文件包含的 key：", list(data.keys()))
print("="*60)

# 查看 state
if "state" in data:
    eef_state = data["state"]
    print("\n✅ state EEF 轨迹：")
    print("   总帧数：", len(eef_state))
    print("   第 0 帧原始数据：")
    print(eef_state[0])
    print("   第 172 帧原始数据：")
    print(eef_state[60])

# 查看 action
if "action" in data:
    eef_action = data["action"]
    print("\n✅ action EEF 轨迹：")
    print("   总帧数：", len(eef_action))
    print("   第 0 帧原始数据：")
    print(eef_action[0])
    print("   第 172 帧原始数据：")
    print(eef_action[60])

print("\n" + "="*60)
print("🧐 判断是不是 EEF：")
print("   - 如果出现 left_eef / right_eef → ✅ 是末端执行器位姿")
print("   - 如果出现 [x,y,z,rx,ry,rz] → ✅ 是3D位置+旋转")
print("="*60)
