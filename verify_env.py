#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境验证脚本 - 确保所有依赖和模块都能正常使用
"""
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("环境配置验证")
print("=" * 60)

try:
    # 验证核心依赖
    print("\n[1] 验证依赖包...")
    import numpy as np
    print(f"    ✓ NumPy {np.__version__}")
    
    import scipy
    print(f"    ✓ SciPy {scipy.__version__}")
    
    from scipy.interpolate import RectBivariateSpline
    from scipy.stats import norm
    print(f"    ✓ SciPy 子模块加载成功")
    
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    print(f"    ✓ SciPy 稀疏矩阵模块加载成功")
    
    # 验证项目模块
    print("\n[2] 验证项目模块...")
    from core.vol_surface import VolSurface
    print("    ✓ core.vol_surface")
    
    from core.dupire import LocalVolPricer
    print("    ✓ core.dupire")
    
    from pricer.pde_pricer import PDEPricer
    print("    ✓ pricer.pde_pricer")
    
    # 快速功能测试
    print("\n[3] 快速功能测试...")
    K_grid = np.linspace(80, 120, 5)
    T_grid = np.linspace(0.1, 1.0, 5)
    iv_matrix = np.array([[0.25, 0.24, 0.23, 0.24, 0.25]] * 5).T
    
    vol_surface = VolSurface(K_grid, T_grid, iv_matrix)
    print("    ✓ VolSurface 初始化成功")
    
    lv_pricer = LocalVolPricer(vol_surface, 100.0, r=0.05, q=0.02)
    print("    ✓ LocalVolPricer 初始化成功")
    
    pde_engine = PDEPricer(lv_pricer, r=0.05, q=0.02)
    print("    ✓ PDEPricer 初始化成功")
    
    # 测试基本计算
    iv = vol_surface.get_iv(100, 0.5)
    print(f"    ✓ 波动率插值正常: IV(K=100, T=0.5) = {iv:.4f}")
    
    call_price = vol_surface.black_scholes_call(100, 100, 0.5, 0.05, 0.02)
    print(f"    ✓ Black-Scholes 定价正常: Call Price = {call_price:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 环境配置完整！所有模块都能正常使用。")
    print("=" * 60)
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
