import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


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
                leverage=10.0,
                maint_margin_rate = 1e-3,
                liquidation_penalty=3,
                mode='train'):
        super().__init__()
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.mode = mode
        self.leverage = leverage
        self.maint_margin_rate = maint_margin_rate
        self.liquidation_penalty = liquidation_penalty

        # action: [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # obs: len(indicators) + 2
        n_obs = len(_INDICATORS) + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.load_data(data_cwd, time_frequency=1)

        self.reset()


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wallet = self.initial_account
        self.pos_amt = 0.0
        self.entry_price = 0

        return self._state(), {
            "env_total_asset": self._equity(0),
            "env_account": self.wallet,
            "env_day_price": self.price_ary[self.t],
            "env_portfolio": self.pos_amt,
            "env_strategy_return": 0,
        }


    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        price = float(self.price_ary[self.t])  # close as mark
        fee_rate = float(self.transaction_fee_percent)

        prev_equity = self._equity(price)

        target_notional = a * self.wallet * self.leverage
        target_pos_amt = target_notional / price
        delta = target_pos_amt - self.pos_amt  # 需要买(+)/卖(-)多少数量
        trade_notional = abs(delta) * price
        fee = trade_notional * fee_rate
        self.wallet -= fee

        if self.pos_amt == 0.0:
            # 开新仓
            self.pos_amt = target_pos_amt
            self.entry_price = price if self.pos_amt != 0 else 0.0
        else:
            same_dir = (self.pos_amt > 0 and target_pos_amt > 0) or (self.pos_amt < 0 and target_pos_amt < 0)
            if same_dir and abs(target_pos_amt) > abs(self.pos_amt):
                # 加仓：更新加权开仓均价
                add_amt = delta
                new_amt = self.pos_amt + add_amt
                self.entry_price = (self.pos_amt * self.entry_price + add_amt * price) / new_amt
                self.pos_amt = new_amt
            else:
                # 减仓 或 反手：先对“被平掉的那部分”结算已实现PnL
                closed_amt = min(abs(self.pos_amt), abs(delta))  # 被关闭的数量
                sign = 1.0 if self.pos_amt > 0 else -1.0
                realized = closed_amt * (price - self.entry_price) * sign
                self.wallet += realized

                # 更新持仓到目标
                self.pos_amt = target_pos_amt
                if self.pos_amt == 0.0:
                    self.entry_price = 0.0
                else:
                    # 反手后剩余的新仓，开仓价按当前成交价记（简化）
                    if not same_dir:
                        self.entry_price = price
                    # 若是减仓后仍同向保留仓位，entry_price 不变

        # move forward
        self.t += 1
        end = (self.t + 1) == self.price_ary.shape[0]

        # === 计算新净值 + 爆仓判定 ===
        price2 = float(self.price_ary[self.t])
        equity = self._equity(price2)

        notional = abs(self.pos_amt) * price2
        maint = notional * self.maint_margin_rate

        liquidated = equity <= maint
        terminated = bool(liquidated)
        truncated = bool(end and not terminated)

        if liquidated:
            self.wallet += self._u_pnl(price2)
            self.pos_amt = 0.0
            self.entry_price = 0.0
        
        reward = np.log10(self._equity(price2) / (prev_equity+1e-6))

        return self._state(), reward, terminated, truncated, {
            "env_total_asset": equity,
            "env_account": self.wallet,
            "env_day_price": self.price_ary[self.t],
            "env_portfolio": self.pos_amt,
            "env_strategy_return": equity / (prev_equity+1e-6) - 1,
        }
    

    def _equity(self, day_price):
        return self.wallet + self._u_pnl(day_price)
    
    def _u_pnl(self, day_price):
        return self.pos_amt * (day_price - self.entry_price)
    
    
    def _state(self):
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
                self.wallet * 2**-18,
                day_price * 2**-15,
                self.entry_price * 2**-15,
                normalized_tech,
                self.pos_amt * 2**-4,
            )
        ).astype(np.float32)
        return state


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
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs