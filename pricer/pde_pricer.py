import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

class PDEPricer:
    def __init__(self, local_vol_pricer, r=0.0, q=0.0):
        """
        :param local_vol_pricer: 我们之前写的 dupire.py 中的 LocalVolPricer 实例
        :param r: 无风险利率
        :param q: 连续分红率
        """
        self.lv_pricer = local_vol_pricer
        self.r = r
        self.q = q

    def price_european(self, S0, K, T, is_call=True):
        """
        使用隐式有限差分法(Implicit FD)给欧式期权定价
        """
        # 1. 搭建网格 (Grid Setup)
        # 几何意义：把“金属棒”(股价空间) 和“时间轴”切分成无数个小格子
        M = 200      # 空间步数 (股价切分多少段)
        N = 200      # 时间步数 (距离到期日切分多少段)
        
        S_max = 3.0 * S0  # 假设股价最高涨到 3 倍
        dS = S_max / M
        dt = T / N
        
        # 构造股价空间网格点 (S_0, S_1, ..., S_M)
        S_grid = np.linspace(0, S_max, M + 1)
        
        # 2. 设置边界条件 (Boundary Conditions)
        # 几何意义：在 T 时刻(到期日)那一瞬间，金属棒上各点的温度(期权内在价值)
        V = np.zeros(M + 1)
        if is_call:
            V = np.maximum(S_grid - K, 0.0)
        else:
            V = np.maximum(K - S_grid, 0.0)

        # 3. 时间倒流，逆向求解 (Time Stepping Backward)
        # 物理图景：从 T 时刻开始，一步一步往回退，计算昨天、前天...直到今天的温度分布
        for n in range(N - 1, -1, -1):
            t_current = n * dt  # 当前所处的时间节点
            
            # 初始化对角线元素
            alpha = np.zeros(M + 1)
            beta = np.zeros(M + 1)
            gamma = np.zeros(M + 1)
            
            # 获取当前时间点 t_current 下，每个空间节点 S_i 的局部波动率
            # 老师敲黑板：这就是 Local Volatility 的精髓！
            # 整个网格的“导热系数”不是常数，而是随位置(S_i)和时间(t_current)剧烈变化的！
            for i in range(1, M):
                # 调用我们之前写的引擎，提取局部波动率
                sigma_lv = self.lv_pricer.local_vol(S_grid[i], t_current)
                
                # BSM PDE 差分项系数配置 (漂移对流 + 波动扩散)
                # 这三个系数决定了热量如何向左(alpha)和向右(gamma)传递，以及原地留存(beta)
                alpha[i] = 0.5 * dt * ((sigma_lv**2) * (i**2) - (self.r - self.q) * i)
                beta[i]  = 1.0 + dt * ((sigma_lv**2) * (i**2) + self.r)
                gamma[i] = 0.5 * dt * ((sigma_lv**2) * (i**2) + (self.r - self.q) * i)
            
            # 处理空间边界条件 (S=0 和 S=S_max 的极端情况)
            beta[0] = 1.0 + self.r * dt
            gamma[0] = 0.0
            alpha[-1] = 0.0
            beta[-1] = 1.0 + self.r * dt  # 为了简化，假设边界二阶导为0 (线性延展)
            
            # 构建三对角矩阵 A
            # 几何意义：A 是一个“时光机矩阵”，满足 A * V_yesterday = V_today
            A = sparse.diags([-alpha[1:], beta, -gamma[:-1]], offsets=[-1, 0, 1], format='csc')
            
            # 求解线性方程组，得到上一个时间步的期权价值向量
            V = spsolve(A, V)

        # 4. 获取今天的最终结果
        # 现在 V 里面装的是今天 (t=0) 时刻，不同股价对应的期权价格。
        # 我们用一维插值，精准提取出对应当前实际股价 S0 的期权价格。
        interpolator = interp1d(S_grid, V, kind='cubic')
        price_today = interpolator(S0)
        
        return price_today