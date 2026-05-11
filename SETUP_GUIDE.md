# Local Vol Pricer 项目使用指南

## 环境配置状态

✅ **环境已配置完毕！**

### 已安装的依赖：
- NumPy 2.4.4 - 数值计算
- SciPy 1.17.1 - 科学计算（插值、优化、求解器等）
- Jupyter - 交互式笔记本
- Pytest - 测试框架
- Matplotlib - 数据可视化

### 虚拟环境位置：
```
.venv/
```

## 快速开始

### 1. 运行环境验证
验证所有模块和依赖都正常工作：
```powershell
.\.venv\Scripts\python.exe verify_env.py
```

### 2. 运行测试
使用 pytest 运行单元测试：
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

或者直接运行测试文件：
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_calibration.py -v
```

### 3. 在 Jupyter 中运行笔记本
启动 Jupyter Lab：
```powershell
.\.venv\Scripts\jupyter.exe lab
```

然后打开 `notebooks/visualization.ipynb`

### 4. 直接运行 Python 脚本
在项目根目录运行任何 Python 脚本：
```powershell
.\.venv\Scripts\python.exe your_script.py
```

## 项目结构说明

```
local-vol-pricer/
├── core/                 # 核心定价模块
│   ├── vol_surface.py   # 隐含波动率曲面 (IV Surface)
│   ├── dupire.py        # Dupire 局部波动率 (Local Vol)
│   └── __init__.py
├── pricer/              # 定价引擎
│   ├── pde_pricer.py    # PDE 有限差分定价器
│   └── __init__.py
├── data/                # 数据存储目录
├── notebooks/           # Jupyter 笔记本
│   └── visualization.ipynb
├── tests/               # 单元测试
│   └── test_calibration.py
├── requirements.txt     # 依赖列表
├── verify_env.py        # 环境验证脚本
└── README.md
```

## 核心模块说明

### VolSurface (core/vol_surface.py)
- 构建隐含波动率曲面
- 提供双变量样条插值功能
- 计算 Black-Scholes 期权价格

使用示例：
```python
from core.vol_surface import VolSurface
import numpy as np

K_grid = np.linspace(80, 120, 5)
T_grid = np.linspace(0.1, 1.0, 5)
iv_matrix = np.array([[0.25, 0.24, 0.23, 0.24, 0.25]] * 5).T

vol_surface = VolSurface(K_grid, T_grid, iv_matrix)
iv = vol_surface.get_iv(100, 0.5)
call_price = vol_surface.black_scholes_call(100, 100, 0.5, 0.05, 0.02)
```

### LocalVolPricer (core/dupire.py)
- 基于 Dupire 公式计算局部波动率
- 使用有限差分方法
- 支持 IV 曲面的数值微分

使用示例：
```python
from core.dupire import LocalVolPricer

lv_pricer = LocalVolPricer(vol_surface, spot=100.0, r=0.05, q=0.02)
local_vol = lv_pricer.local_vol(K=100, T=0.5)
```

### PDEPricer (pricer/pde_pricer.py)
- 使用隐式有限差分法求解 PDE
- 计算欧式期权价格
- 支持 call 和 put 期权

使用示例：
```python
from pricer.pde_pricer import PDEPricer

pde_engine = PDEPricer(lv_pricer, r=0.05, q=0.02)
price = pde_engine.price_european(S0=100, K=100, T=0.5, is_call=True)
```

## 常见命令速查表

| 任务 | 命令 |
|------|------|
| 验证环境 | `.\.venv\Scripts\python.exe verify_env.py` |
| 运行测试 | `.\.venv\Scripts\python.exe -m pytest tests/ -v` |
| 启动 Jupyter | `.\.venv\Scripts\jupyter.exe lab` |
| 安装新包 | `.\.venv\Scripts\pip.exe install package_name` |
| 升级包 | `.\.venv\Scripts\pip.exe install --upgrade package_name` |

## 故障排除

### 问题：模块找不到 (ModuleNotFoundError)
**解决方案：** 确保从项目根目录运行脚本，虚拟环境已激活。

### 问题：脚本运行速度慢
**解决方案：** PDE 定价器和 Dupire 计算涉及大量数值运算，这是正常的。可以减少网格点数 (M, N) 来加速。

### 问题：导入错误
**解决方案：** 运行 `verify_env.py` 验证环境。如果仍有错误，检查虚拟环境中的包是否完整。

## 需要更新依赖？

编辑 `requirements.txt` 后重新安装：
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt --upgrade
```

## 进一步开发

- 在 `core/` 中添加更多的定价模型
- 在 `pricer/` 中实现美式期权定价
- 在 `data/` 中集成真实市场数据
- 在 `notebooks/` 中创建分析笔记本
- 在 `tests/` 中增加更多单元测试

---

**最后验证时间**: 2026年5月11日  
**Python 版本**: 3.11.9  
**环境类型**: 虚拟环境 (venv)
