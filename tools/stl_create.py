import trimesh

# ===================================================================
#                       请在这里修改参数
# ===================================================================

# 1. 设置球心坐标 (x, y, z)
SPHERE_CENTER = [0.5, 0.5, 0.5]

# 2. 设置球体半径
SPHERE_RADIUS = 0.2

# 3. 设置球体精细度 (细分次数)
#    0 = 20个面 (二十面体)
#    1 = 80个面
#    2 = 320个面
#    3 = 1280个面 (默认，效果不错)
#    4 = 5120个面 (非常平滑)
SUBDIVISIONS = 2

# 4. 设置输出文件名
OUTPUT_FILENAME = "../run/sphere.stl"

# ===================================================================
#                           脚本主逻辑
# ===================================================================

def generate_sphere():
    """
    根据顶部定义的参数创建并保存一个球体STL文件。
    """
    print("--- 正在生成球体 ---")
    print(f"中心:       {SPHERE_CENTER}")
    print(f"半径:       {SPHERE_RADIUS}")
    print(f"细分等级:   {SUBDIVISIONS}")
    print(f"输出文件:   {OUTPUT_FILENAME}")

    # 使用 trimesh 创建一个以原点为中心的二十面体球 (icosphere)
    # 这是生成高质量球体网格的最佳方法
    sphere_mesh = trimesh.creation.icosphere(
        subdivisions=SUBDIVISIONS,
        radius=SPHERE_RADIUS
    )

    # 将球体平移到指定中心
    sphere_mesh.apply_translation(SPHERE_CENTER)

    # 导出为STL文件
    try:
        sphere_mesh.export(file_obj=OUTPUT_FILENAME)
        print("\n成功!")
        print(f"已创建包含 {len(sphere_mesh.vertices)} 个顶点和 {len(sphere_mesh.faces)} 个面的STL文件。")
    except Exception as e:
        print(f"\n错误: 文件保存失败。 {e}")

# 当直接运行此脚本时，执行生成函数
if __name__ == '__main__':
    generate_sphere()