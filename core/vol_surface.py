import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

class VolSurface:
    def __init__(self, strikes, maturities, iv_matrix):
        """
        初始化隐含波动率(IV)曲面
        :param strikes: 1D array, 行权价网格 (K)
        :param maturities: 1D array, 到期时间网格 (T)
        :param iv_matrix: 2D array, 对应的隐含波动率矩阵, shape = (len(strikes), len(maturities))
        """
        self.K_grid = strikes
        self.T_grid = maturities
        
        # 老师敲黑板：这里我们用双变量样条插值(Bivariate Spline)。
        # 几何意义：在给定的 (K, T) 离散点上，拉伸一张平滑且连续的二维曲面。
        # 这样我们在任意非节点的 (K, T) 处，都能取到平滑的波动率。
        self.spline = RectBivariateSpline(self.K_grid, self.T_grid, iv_matrix)

    def get_iv(self, K, T):
        """
        获取任意给定 K 和 T 处的平滑隐含波动率
        """
        # 注意：RectBivariateSpline 返回的是二维数组，我们需要用 [0,0] 提取标量
        return self.spline(K, T)[0, 0]

    def black_scholes_call(self, S, K, T, r, q=0):
        """
        根据平滑后的 IV 曲面，反算 Black-Scholes 看涨期权价格
        几何意义：我们用平滑的“橡胶皮”重新生成了毫无噪音的理论期权价格表面
        """
        # 如果时间 T 极小，期权价格趋近于内在价值
        if T <= 1e-5:
            return max(S - K, 0.0)
            
        sigma = self.get_iv(K, T)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price