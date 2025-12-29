from __future__ import annotations

import numpy as np
import gymnasium as gym
import pandas as pd
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

class BitcoinEnv(gym.Env):  # custom env
    def __init__(
        self,
        data_cwd=None,
        time_frequency=1,
        initial_account=1e6,
        transaction_fee_percent=1e-3,
        mode="train",
        gamma=0.99,
    ):
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.gamma = gamma
        self.mode = mode
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, ))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(_INDICATORS)+3, )
        )
        self.load_data(
            data_cwd, time_frequency
        )

        # reset
        self.day = 0
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        self.stocks = 0.0  # multi-stack

        self.total_asset = self.account + self.day_price[0] * self.stocks
        self.episode_return = 0.0
        self.gamma_return = 0.0

        """env information"""
        self.env_name = "BitcoinEnv4"
        self.state_dim = 1 + 1 + self.price_ary.shape[1] + self.tech_ary.shape[1]
        self.action_dim = 1
        self.if_discrete = False
        self.target_return = 10
        self.max_step = self.price_ary.shape[0]

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> np.ndarray:
        self.day = 0
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        self.initial_account__reset = self.initial_account  # reset()
        self.account = self.initial_account__reset
        self.stocks = 0.0
        self.total_asset = self.account + self.day_price[0] * self.stocks

        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]
        state = np.hstack(
            (
                self.account * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)
        return state, {
            "env_total_asset": self.total_asset,
            "env_account": self.account,
            "env_day_price": self.day_price,
            "env_portfolio": self.stocks,
            "env_strategy_return": 0,
        }

    def step(self, action):
        prev_total_asset = self.total_asset
        stock_action = action[0]
        """buy or sell stock"""
        adj = self.day_price[0]
        if stock_action > 0:
            max_buy_units = self.account / (adj * (1.0 + self.transaction_fee_percent) + 1e-12)
            buy_units = stock_action * max_buy_units
            cost = buy_units * adj * (1.0 + self.transaction_fee_percent)
            self.account -= cost
            self.stocks += buy_units
        else:
            sell_units = (-stock_action) * self.stocks
            revenue = sell_units * adj * (1.0 - self.transaction_fee_percent)
            self.account += revenue
            self.stocks -= sell_units

        """update day"""
        self.day += 1
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        done = (self.day + 1) == self.max_step
        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]
        state = np.hstack(
            (
                self.account * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)

        next_total_asset = self.account + self.day_price[0] * self.stocks
        reward = (next_total_asset / self.total_asset - 1) * 100
        self.total_asset = next_total_asset

        return state, reward, done, False, {
            "env_total_asset": self.total_asset,
            "env_account": self.account,
            "env_day_price": self.day_price,
            "env_portfolio": self.stocks,
            "env_strategy_return": (next_total_asset / prev_total_asset - 1) * 100,
        }

    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()
        episode_returns.append(1)
        btc_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.max_step):
                if i == 0:
                    init_price = self.day_price[0]
                btc_returns.append(self.day_price[i] / init_price)
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                episode_returns.append(self.total_asset / 1e6)
                if done:
                    break

        import matplotlib.pyplot as plt

        plt.plot(episode_returns, label="agent return")
        plt.plot(btc_returns, color="yellow", label="BTC return")
        plt.grid()
        plt.title("cumulative return")
        plt.xlabel("day")
        plt.xlabel("multiple of initial_account")
        plt.legend()
        plt.savefig(f"{cwd}/cumulative_return.jpg")
        return episode_returns, btc_returns

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



class BitcoinEnvDSR(BitcoinEnv):
    """
    用 Differential Sharpe Ratio (DSR) 做逐步reward，近似最大化Sharpe。
    """

    def __init__(
        self,
        *args,
        rf_per_step: float = 0.0,      # 每步无风险对数收益（可先设0）
        dsr_eta: float = 0.05,         # EMA步长；约等于 1/window
        dsr_eps: float = 1e-6,         # 数值稳定
        dsr_reward_scale: float = 1.0, # DSR reward缩放，便于PPO稳定训练
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rf_per_step = float(rf_per_step)
        self.dsr_eta = float(dsr_eta)
        self.dsr_eps = float(dsr_eps)
        self.dsr_reward_scale = float(dsr_reward_scale)

        # DSR内部状态：A=E[x], B=E[x^2] 的EMA估计
        self._A = 0.0
        self._B = 0.0

        # 如果要把DSR统计量拼进state，需要扩展observation_space
        base_dim = self.observation_space.shape[0]
        # 这里拼两个量：ema_mean(%), ema_std(%)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + 2,), dtype=np.float32
        )

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        var = max(self._B - self._A * self._A, self.dsr_eps)
        std = np.sqrt(var)

        # 为了量级接近你原 reward*100 的风格，这里乘100变成“百分比量级”
        extra = np.array([self._A, std], dtype=np.float32)
        return np.concatenate([state.astype(np.float32), extra], axis=0)

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)

        # reset DSR moments
        self._A = 0.0
        self._B = 0.0

        state = self._augment_state(state)
        return state, info

    def step(self, action):
        prev_total_asset = float(self.total_asset)
        A_prev, B_prev = float(self._A), float(self._B)

        # 先让父类完成交易、推进时间、更新 total_asset，并给出原始reward
        state, raw_reward, terminated, truncated, info = super().step(action)

        # （prev_total_asset应>0；加eps避免log(0)）
        # ratio = max(float(self.total_asset) / max(prev_total_asset, 1e-12), 1e-12)
        # x_t = np.log(ratio) - self.rf_per_step
        x_t = (self.total_asset / prev_total_asset - 1) * 100

        # DSR 增量
        eta = self.dsr_eta
        dA = eta * (x_t - A_prev)
        dB = eta * (x_t * x_t - B_prev)

        var_prev = max(B_prev - A_prev * A_prev, self.dsr_eps)
        denom = (var_prev ** 1.5)

        dsr_reward = 0.0
        if denom > 0:
            dsr_reward = (B_prev * dA - 0.5 * A_prev * dB) / denom

        # 更新 moments
        self._A = A_prev + dA
        self._B = B_prev + dB

        # 构造最终 reward
        reward = float(self.dsr_reward_scale * dsr_reward)

        # 更新 next state（把DSR统计量拼进去）
        state = self._augment_state(state)

        return state, reward, terminated, truncated, info