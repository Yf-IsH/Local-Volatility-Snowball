import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径，使得可以导入 core 和 pricer 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 假设我们导入了之前写好的三大核心模块
from core.vol_surface import VolSurface
from core.dupire import LocalVolPricer
from pricer.pde_pricer import PDEPricer

def test_local_vol_calibration():
    print("开始执行 Local Volatility 模型一致性检验 (Sanity Check)...\n")
    
    # 1. 初始化市场环境假设 (在实际项目中，这里应该从 data/ 目录读取真实的 IV 数据)
    S0 = 100.0
    r = 0.05
    q = 0.02
    
    # 模拟一个简单的平滑隐含波动率网格 (仅作测试演示)
    # 真实项目中你会传入真正的平滑 IV 矩阵
    K_grid = np.linspace(80, 120, 5)
    T_grid = np.linspace(0.1, 1.0, 5)
    # 假设一个自带 Volatility Smile (波动率微笑) 的假数据
    iv_matrix = np.array([[0.25, 0.24, 0.23, 0.24, 0.25]] * 5).T 
    
    # 2. 组装流水线 (Pipeline)
    print("-> 正在构建隐含波动率曲面...")
    vol_surface = VolSurface(K_grid, T_grid, iv_matrix)
    
    print("-> 正在实例化 Dupire 局部波动率引擎...")
    lv_pricer = LocalVolPricer(vol_surface, S0, r, q)
    
    print("-> 正在启动 PDE 偏微分方程求解器...\n")
    pde_engine = PDEPricer(lv_pricer, r, q)
    
    # 3. 核心检验环节：抽查几个关键的 (K, T) 节点
    test_points = [
        {"K": 90.0, "T": 0.5, "type": "Call (实值)"},
        {"K": 100.0, "T": 0.5, "type": "Call (平值)"},
        {"K": 110.0, "T": 1.0, "type": "Call (虚值)"}
    ]
    
    tolerance = 1e-2  # 业界通常允许极小的数值离散误差 (比如1美分以内)
    passed_all = True

    print(f"{'期权类型':<15} | {'目标行权价(K)':<12} | {'到期时间(T)':<10} | {'市场解析解(BS)':<15} | {'PDE倒推解(LV)':<15} | {'误差(Error)'}")
    print("-" * 100)
    
    for pt in test_points:
        K_test = pt["K"]
        T_test = pt["T"]
        
        # [基准]：直接用平滑后的 IV 代入 Black-Scholes 公式算出的“目标真理价格”
        target_price = vol_surface.black_scholes_call(S0, K_test, T_test, r, q)
        
        # [测试]：用 PDE 挂载 Local Volatility 算出的“模型倒推价格”
        pde_price = pde_engine.price_european(S0, K_test, T_test, is_call=True)
        
        # 计算绝对误差
        error = abs(target_price - pde_price)
        
        print(f"{pt['type']:<13} | K={K_test:<10} | T={T_test:<8} | {target_price:<16.4f} | {pde_price:<16.4f} | {error:.6f}")
        
        if error > tolerance:
            passed_all = False

    print("-" * 100)
    # 4. 给出最终结论
    if passed_all:
        print("\n✅ 检验通过！PDE 引擎搭配局部波动率完美复刻了市场基准价格。模型闭环逻辑正确！")
        print("下一步：你可以放心地用这套引擎去给带有障碍 (Barrier) 或美式 (American) 特征的奇异期权定价了。")
    else:
        print("\n❌ 检验失败！误差过大。")
        print("请回头检查：\n1. dupire.py 中的差分算子是否写错？\n2. pde_pricer.py 的边界条件 (Boundary Conditions) 或网格步长 (dt, dS) 是否足够精细？")

# 运行测试
if __name__ == "__main__":
    test_local_vol_calibration()