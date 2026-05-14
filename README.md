# Local Volatility Snowball Coupon Demo

一个用于演示 local volatility 如何服务于雪球票息报价的 Python demo。

## 功能

- 接入东方财富日线行情，支持 ETF 与宽基指数标的。
- 对 ETF 标的接入上交所期权风险指标 `IMPLC_VOLATLTY`，构建真实 IV 曲面。
- 使用 Dupire 公式从 IV 曲面生成 local volatility surface。
- 用 local vol Monte Carlo 模拟雪球路径，并反解 `PV(coupon) = notional` 的公允年化票息。
- 前端展示可交互 3D IV 曲面、local vol 曲面、路径四分类概率和详细数学推导。

## 运行

```powershell
.\.venv\Scripts\python.exe demo_app.py --port 8015
```

浏览器打开：

```text
http://127.0.0.1:8015
```

如果端口被占用，程序会自动顺延到下一个可用端口，并在终端打印实际地址。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 数据说明

ETF 期权 IV 来自上交所期权风险指标接口中的 `IMPLC_VOLATLTY` 字段。程序会过滤 0 或极端 IV，并把合约点插值成规则曲面。网络不可用时，行情和 IV 会尝试读 `data/cache/`，缓存目录不提交到 Git。

宽基指数如中证500、中证1000当前仍使用历史波动率生成的演示 IV 曲面；若要生产级使用，应进一步接入中金所股指期权链并反推 IV。
