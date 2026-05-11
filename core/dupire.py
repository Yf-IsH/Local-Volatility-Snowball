import numpy as np

class LocalVolPricer:
    def __init__(self, vol_surface, spot, r=0.0, q=0.0):
        """
        :param vol_surface: 我们在 vol_surface.py 中构建好的平滑曲面对象
        :param spot: 当前股票现价 (S_0)
        :param r: 无风险利率
        :param q: 连续分红率
        """
        self.vol_surface = vol_surface
        self.S = spot
        self.r = r
        self.q = q

    def local_vol(self, K, T):
        """
        利用 Dupire 公式计算在特定 (K, T) 点的局部波动率
        """
        # 设置差分的步长 (极小的值，用来逼近微积分里的极限)
        dT = 1e-4  # 时间步长
        dK = 1e-2  # 行权价步长 (空间步长)

        # 1. 准备计算所需的基准点和偏移点的期权价格
        # 这些价格都是从我们“无噪音”的平滑 IV 曲面中提取出来的
        C_base = self.vol_surface.black_scholes_call(self.S, K, T, self.r, self.q)
        
        # 时间向前推移一点点
        C_up_T = self.vol_surface.black_scholes_call(self.S, K, T + dT, self.r, self.q)
        
        # 空间(行权价)向左向右偏移一点点
        C_up_K = self.vol_surface.black_scholes_call(self.S, K + dK, T, self.r, self.q)
        C_dn_K = self.vol_surface.black_scholes_call(self.S, K - dK, T, self.r, self.q)

        # 2. 计算分子：时间价值的膨胀率 + 漂移项调整
        # 几何意义：期权价格随时间流逝的“生长速度” (一阶导数)
        dC_dT = (C_up_T - C_base) / dT
        dC_dK = (C_up_K - C_dn_K) / (2 * dK) # 中心差分算一阶导
        
        # 分子项：加入了利率 r 和分红 q 后的完整形式
        numerator = dC_dT + (self.r - self.q) * K * dC_dK + self.q * C_base

        # 3. 计算分母：空间曲率 (蝶式期权)
        # 几何意义：在行权价 K 处的聚集度。也就是 (C_up - 2*C_base + C_dn) / dK^2
        # 这是典型的二阶中心差分公式
        d2C_dK2 = (C_up_K - 2 * C_base + C_dn_K) / (dK ** 2)
        
        denominator = 0.5 * (K ** 2) * d2C_dK2

        # 4. 容错处理与计算局部波动率
        # 老师敲黑板：尽管我们平滑了曲面，但在极端深实值/虚值区域，分母仍可能极小甚至小于0
        if denominator <= 1e-8 or numerator <= 0:
            # 如果出现违背无套利原则的情况，退化返回局部的隐含波动率
            return self.vol_surface.get_iv(K, T)

        # 核心物理图景：局部导热系数 = 扩散速度 / 空间聚集度
        local_var = numerator / denominator
        
        return np.sqrt(local_var)