import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces


_INDICATORS = [
    'macd',          # MACD (默认 12, 26)
    'boll_ub',       # 布林带上轨 (默认 20)
    'boll_lb',       # 布林带下轨 (默认 20)
    'rsi_30',        # 30周期 RSI
    'dx_30',         # 30周期 DX (动向指标)
    'close_30_sma',  # 30周期收盘价简单移动平均
    'close_60_sma'   # 60周期收盘价简单移动平均
]

class USDMEnv(gym.Env):
    """
    Spot-style (equity = cash + qty*close), all pricing & PnL use close.
    obs = tech indicators + [pos_value_ratio, cash_ratio]
    action in [-1, 1]: buy/sell fraction.
    """

    def __init__(self,
                data_cwd=None,
                initial_account=1e6,
                transaction_fee_percent=1e-3,
                mode='train'):
        super().__init__()
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.mode = mode

        # action: [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # obs: len(indicators) + 2
        n_obs = len(_INDICATORS) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.load_data(data_cwd, time_frequency=1)

        self.reset()


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_account
        self.shares = 0.0
        self.entry_price = 0

        day_price = self.price_ary[self.t]
        day_tech =self.tech_ary[self.t]
        normalized_tech = [
            day_tech[0] * 2**-1,
            day_tech[1] * 2**-15,
            day_tech[2] * 2**-15,
            day_tech[3] * 2**-6,
            day_tech[4] * 2**-6,
            day_tech[5] * 2**-15,
            day_tech[6] * 2**-15,
        ]
        
        state = np.hstack(
            (
                self.cash * 2**-18,
                day_price * 2**-15,
                self.entry_price * 2**-15,
                normalized_tech,
                self.shares * 2**-4,
            )
        ).astype(np.float32)

        return state, {
            
        }


    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        day_price = self.price_ary[self.t]

        equity_before = self._equity()

        if a > 0:  # buy with fraction of cash
            spend = a * self.cash
            # 购买时现金支出包含手续费：spend_total = spend*(1+fee)
            # 为不超支：用 spend/(1+fee) 去买币
            buy_cash = spend / (1.0 + self.fee)
            buy_qty = buy_cash / p

            self.cash -= buy_cash * (1.0 + self.fee)
            self.qty += buy_qty

        elif a < 0:  # sell fraction of holdings
            sell_qty = (-a) * self.qty
            revenue = sell_qty * p * (1.0 - self.fee)

            self.qty -= sell_qty
            self.cash += revenue

        # move forward
        self.t += 1
        terminated = False
        truncated = (self.t >= len(self.df) - 1)

        equity_after = self._equity()

        # reward：净值变化（也可以改成 log return）
        reward = float(equity_after - equity_before)

        info = {
            "price": p,
            "equity": equity_after,
            "cash": float(self.cash),
            "positionAmt": float(self.qty),
            "positionValue": float(self.qty * p),
        }
        return self._obs(), reward, terminated, truncated, info
    

    def _equity(self, day_price):
        return self.cash + (day_price - self.entry_price) * self.shares
    




    def load_data(
        self, data_cwd, time_frequency,
    ):
        if self.mode == "train":
            df = pd.read_csv(f"{data_cwd}/train_data.csv")
            self.price_ary = df['close'].to_numpy().reshape(-1, 1)
            self.tech_ary = df[_INDICATORS].to_numpy()
            n = self.price_ary.shape[0]
            x = n // int(time_frequency)
            ind = [int(time_frequency) * i for i in range(x)]
            self.price_ary = self.price_ary[ind]
            self.tech_ary = self.tech_ary[ind]
        else:
            df = pd.read_csv(f"{data_cwd}/trade_data.csv")
            self.price_ary = df['close'].to_numpy().reshape(-1, 1)
            self.tech_ary = df[_INDICATORS].to_numpy()
            n = self.price_ary.shape[0]
            x = n // int(time_frequency)
            ind = [int(time_frequency) * i for i in range(x)]
            self.price_ary = self.price_ary[ind]
            self.tech_ary = self.tech_ary[ind]