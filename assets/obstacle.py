"""
Script to generate standalone USD assets for obstacles.
Compatible with Isaac Lab (formerly Orbit).
Usage: python generate_obstacles.py
"""

import argparse
# [修正] 使用新的 Isaac Lab 包名
from isaaclab.app import AppLauncher

# 1. 启动 Isaac Sim 应用 (必须在导入 omni 之前)
parser = argparse.ArgumentParser(description="Generate Obstacle Assets")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入必要的 Omniverse 模块
import os
# 注意：pxr 是 Isaac Sim 内置的 USD 库，不需要改为 isaaclab
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt

# ==============================================================================
# 配置区域
# ==============================================================================
OUTPUT_DIR = "/home/wei/End_to_end/isaac_drone/assets/obstacle"
WIDTHS = [0.25, 0.50, 0.75, 1.00]

# 颜色定义 (RGB)
COLOR_GREEN = (0.0, 1.0, 0.0) # Cuboid
COLOR_RED   = (1.0, 0.0, 0.0) # Cylinder

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory: {path}")

def save_stage(stage, path):
    # Stage 创建时已经指定了路径，直接无参调用 Save() 即可
    stage.GetRootLayer().Save() 
    print(f"[SUCCESS] Saved asset to: {path}")

def create_basic_stage(path):
    """创建一个新的空 Stage"""
    if os.path.exists(path):
        os.remove(path)
    
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # 创建默认的 Prim 根节点 (Xform)
    # 这一步很重要，确保导入时有一个根节点方便移动
    root_prim = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root_prim.GetPrim())
    return stage, root_prim

def apply_physics_and_material(prim, stage, color_rgb):
    """添加刚体属性、碰撞属性和颜色"""
    # 1. 添加碰撞 API
    UsdPhysics.CollisionAPI.Apply(prim)
    
    # 2. 添加刚体 API (Rigid Body)
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    # 默认设为 Kinematic (运动学)，因为我们在环境中要手动控制位置
    rb_api.CreateKinematicEnabledAttr(True) 

    # 3. 设置颜色 (DisplayColor)
    gprim = UsdGeom.Gprim(prim)
    # 使用 Vt.Vec3fArray 确保类型匹配
    color_array = Vt.Vec3fArray([Gf.Vec3f(*color_rgb)])
    gprim.GetDisplayColorAttr().Set(color_array)

def generate_cuboids():
    """生成 4 种长方体"""
    height = 1.0
    
    for w in WIDTHS:
        file_name = f"cuboid_{w:.2f}.usd"
        full_path = os.path.join(OUTPUT_DIR, file_name)
        
        stage, root = create_basic_stage(full_path)
        
        # 创建 Cube
        cube_path = root.GetPath().AppendChild("Geometry")
        cube = UsdGeom.Cube.Define(stage, cube_path)
        
        # 设置尺寸
        # UsdGeom.Cube 默认边长是 2.0 (范围 -1 到 1)
        # 目标: x=w, y=w, z=height
        # 缩放因子 = 目标尺寸 / 2.0
        scale = Gf.Vec3d(w/2.0, w/2.0, height/2.0)
        
        # 应用变换
        xform = UsdGeom.Xformable(cube)
        xform.AddScaleOp().Set(scale)
        # 默认 Cube 中心在 (0,0,0)
        
        # 添加物理和颜色
        apply_physics_and_material(cube.GetPrim(), stage, COLOR_GREEN)
        
        save_stage(stage, full_path)

def generate_cylinders():
    """生成 4 种圆柱体"""
    height = 5.0
    
    for w in WIDTHS:
        radius = w / 2.0
        file_name = f"cylinder_{w:.2f}.usd"
        full_path = os.path.join(OUTPUT_DIR, file_name)
        
        stage, root = create_basic_stage(full_path)
        
        # 创建 Cylinder
        cyl_path = root.GetPath().AppendChild("Geometry")
        cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
        
        # 设置属性
        cyl.CreateRadiusAttr(radius)
        cyl.CreateHeightAttr(height)
        # 设置轴向 (Z轴朝上)
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        # 默认 Cylinder 中心在 (0,0,0)，也就是一半在地下一半在地上
        # 这对于环境放置不太方便，我们通常希望它的原点在几何中心
        # Isaac Sim 的 Cylinder 默认就是中心点为原点，所以不需要额外移动
        
        # 添加物理和颜色
        apply_physics_and_material(cyl.GetPrim(), stage, COLOR_RED)
        
        save_stage(stage, full_path)

def main():
    print(f"[INFO] Starting Asset Generation...")
    print(f"[INFO] Output Directory: {OUTPUT_DIR}")
    
    create_folder(OUTPUT_DIR)
    
    generate_cuboids()
    generate_cylinders()
    
    print("[INFO] Generation Complete!")

if __name__ == "__main__":
    main()
    simulation_app.close()