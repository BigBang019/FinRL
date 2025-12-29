import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from stable_baselines3.common.callbacks import EventCallback, BaseCallback, EvalCallback


class TradeCallback(EvalCallback):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "explained_variance": [],
            "total_return": [],
            "sharpe_ratio": []
        }
        self.current_epoch_info = {}
    
    @torch.no_grad()
    def _on_step(self):
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.model.policy.eval()
            obs = self.eval_env.reset()

            current_obs = obs
            current_reward = np.zeros(1, )

            collected_obs = []
            collected_actions = []
            collected_values = [] # 如果需要计算 Value Loss

            collected_total_asset = []
            collected_portfolio = []
            collected_strategy_return = []
            collected_day_price = []
            
            while True:
                action, state = self.model.predict(current_obs, deterministic=True)
                obs_tensor = torch.as_tensor(current_obs).to(self.model.device)
                value = self.model.policy.predict_values(obs_tensor)

                collected_obs.append(obs_tensor)
                collected_actions.append(torch.as_tensor(action).to(self.model.device))
                collected_values.append(value)

                next_obs, rewards, dones, infos = self.eval_env.step(action)
                current_reward += rewards

                info = infos[0]
                collected_total_asset.append(info['env_total_asset'])
                collected_portfolio.append(info['env_portfolio'])
                collected_strategy_return.append(info['env_strategy_return'])
                collected_day_price.append(info['env_day_price'])

                current_obs = next_obs
                if dones.any(): # 假设只有一个env
                    break
            
            obs_tensor = torch.cat(collected_obs)
            actions_tensor = torch.cat(collected_actions)
            values_tensor = torch.cat(collected_values).flatten()

            values, log_prob, entropy = self.model.policy.evaluate_actions(obs_tensor, actions_tensor)

            entropy_loss = -torch.mean(entropy)

            self.metrics['entropy_loss'].append(entropy_loss.item())
            
            total_asset_array = np.array(collected_total_asset) # (n, )
            portfolio_array = np.array(collected_portfolio) # (n, )
            strategy_return_array = np.array(collected_strategy_return) # (n, )
            day_price_array = np.array(collected_day_price) # (n, )

            self.current_epoch_info['total_asset_array'] = total_asset_array
            self.current_epoch_info['portfolio_array'] = portfolio_array
            self.current_epoch_info['strategy_return_array'] = strategy_return_array
            self.current_epoch_info['day_price_array'] = day_price_array
            self.current_epoch_info['action_array'] = actions_tensor.detach().cpu().numpy()
            
            total_return = (total_asset_array[-1] / total_asset_array[0] - 1) * 100
            self.metrics['total_return'].append(total_return)
            
            sharpe_ratio = np.mean(strategy_return_array) / np.std(strategy_return_array)
            self.metrics['sharpe_ratio'].append(sharpe_ratio)
            
            self.logger.record(f"{self.name}/entropy_loss", entropy_loss.item())
            self.logger.record(f"{self.name}/total_return", float(total_return))
            self.logger.record(f"{self.name}/sharpe_ratio", float(sharpe_ratio))

            mean_reward = np.mean(current_reward)
            self.last_mean_reward = mean_reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            self.logger.record(f"{self.name}/mean_reward", float(mean_reward))
            self.plot_current_epoch(save_path=f"{self.model.logger.get_dir()}/{self.name}_{self.n_calls}.png")
            
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def plot(self, save_path=None, figsize=(15, 10)):
        """
        绘制self.metrics和self.current_epoch_info的可视化图表
        
        Args:
            save_path: 保存图片的路径，如果为None则不保存
            figsize: 图片大小，默认(15, 10)
        """
        if not self.metrics or len(self.metrics['total_return']) == 0:
            print("Warning: No metrics data to plot")
            return
        
        # 创建图形
        fig = plt.figure(figsize=figsize)
        
        # 准备x轴数据（评估次数）
        eval_steps = list(range(len(self.metrics['total_return'])))
        
        # 1. 绘制metrics历史趋势（2x3布局）
        # 1.1 Total Return
        ax1 = plt.subplot(2, 3, 1)
        if len(self.metrics['total_return']) > 0:
            plt.plot(eval_steps, self.metrics['total_return'], 'b-', linewidth=2, label='Total Return (%)')
            plt.xlabel('Evaluation Step')
            plt.ylabel('Total Return (%)')
            plt.title(f'{self.name} - Total Return')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 1.2 Sharpe Ratio
        ax2 = plt.subplot(2, 3, 2)
        if len(self.metrics['sharpe_ratio']) > 0:
            plt.plot(eval_steps, self.metrics['sharpe_ratio'], 'g-', linewidth=2, label='Sharpe Ratio')
            plt.xlabel('Evaluation Step')
            plt.ylabel('Sharpe Ratio')
            plt.title(f'{self.name} - Sharpe Ratio')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 1.3 Entropy Loss
        ax3 = plt.subplot(2, 3, 3)
        if len(self.metrics['entropy_loss']) > 0:
            plt.plot(eval_steps, self.metrics['entropy_loss'], 'r-', linewidth=2, label='Entropy Loss')
            plt.xlabel('Evaluation Step')
            plt.ylabel('Entropy Loss')
            plt.title(f'{self.name} - Entropy Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 1.4 Total Return vs Sharpe Ratio (双y轴)
        ax4 = plt.subplot(2, 3, 4)
        if len(self.metrics['total_return']) > 0 and len(self.metrics['sharpe_ratio']) > 0:
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(eval_steps, self.metrics['total_return'], 'b-', linewidth=2, label='Total Return (%)')
            line2 = ax4_twin.plot(eval_steps, self.metrics['sharpe_ratio'], 'g-', linewidth=2, label='Sharpe Ratio')
            ax4.set_xlabel('Evaluation Step')
            ax4.set_ylabel('Total Return (%)', color='b')
            ax4_twin.set_ylabel('Sharpe Ratio', color='g')
            ax4.tick_params(axis='y', labelcolor='b')
            ax4_twin.tick_params(axis='y', labelcolor='g')
            plt.title(f'{self.name} - Return vs Sharpe')
            ax4.grid(True, alpha=0.3)
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
        
        # 1.5 最新评估的资产曲线
        ax5 = plt.subplot(2, 3, 5)
        if self.current_epoch_info and 'total_asset_array' in self.current_epoch_info:
            total_asset = self.current_epoch_info['total_asset_array']
            time_steps = np.arange(len(total_asset))
            plt.plot(time_steps, total_asset, 'b-', linewidth=2, label='Total Asset')
            plt.xlabel('Time Step')
            plt.ylabel('Total Asset')
            plt.title(f'{self.name} - Latest Episode Asset Curve')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 1.6 最新评估的收益率分布
        ax6 = plt.subplot(2, 3, 6)
        if self.current_epoch_info and 'strategy_return_array' in self.current_epoch_info:
            returns = self.current_epoch_info['strategy_return_array']
            plt.hist(returns, bins=30, alpha=0.7, color='purple', edgecolor='black')
            plt.xlabel('Return (%)')
            plt.ylabel('Frequency')
            plt.title(f'{self.name} - Latest Episode Return Distribution')
            plt.axvline(np.mean(returns), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
            plt.axvline(np.median(returns), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.2f}%')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_current_epoch(self, save_path=None, figsize=(14, 10)):
        """
        绘制当前epoch的详细信息（含 action 曲线 + 持仓占总资产百分比）

        数据含义（按你描述）：
        - total_asset_array: 每步总资产（含现金+持仓市值）
        - portfolio_array: 每步持仓股数
        - day_price_array: 每步股票单价
        - strategy_return_array: 每步策略收益（百分比）
        - action_array: [-1, 1] 表示卖出/买入百分比（标量动作）
        """
        if not self.current_epoch_info:
            print("Warning: No current epoch info to plot")
            return

        # ---------- 取数据并对齐长度 ----------
        total_asset = np.asarray(self.current_epoch_info.get("total_asset_array", []), dtype=float)
        shares = np.asarray(self.current_epoch_info.get("portfolio_array", []), dtype=float)
        price = np.asarray(self.current_epoch_info.get("day_price_array", []), dtype=float)
        returns = np.asarray(self.current_epoch_info.get("strategy_return_array", []), dtype=float)
        actions = np.asarray(self.current_epoch_info.get("action_array", []), dtype=float)

        if total_asset.size == 0:
            print("Warning: total_asset_array is empty")
            return

        T = total_asset.shape[0]
        t = np.arange(T)
        eps = 1e-12

        # ---------- 计算：持仓市值 & 持仓占比 ----------
        # 支持单标的：(T,) * (T,)
        # 也支持多标的：(T, n_assets) * (T, n_assets) -> sum over assets
        if shares.ndim == 1 and price.ndim == 1:
            holding_value = shares * price
        elif shares.ndim == 2 and price.ndim == 2:
            holding_value = np.sum(shares * price, axis=1)
        elif shares.ndim == 2 and price.ndim == 1:
            holding_value = np.sum(shares * price[:, None], axis=1)
        elif shares.ndim == 1 and price.ndim == 2:
            holding_value = np.sum(shares[:, None] * price, axis=1)
        else:
            # 兜底：尽量拉平对齐
            holding_value = np.asarray(shares, dtype=float).reshape(T, -1)
            holding_value = np.sum(holding_value, axis=1)

        holding_pct = np.where(np.abs(total_asset) > eps, holding_value / total_asset * 100.0, 0.0)

        # 保存到 current_epoch_info，方便你在外面直接取
        self.current_epoch_info["holding_value_array"] = holding_value
        self.current_epoch_info["holding_pct_array"] = holding_pct

        # ---------- 处理 action 形状（你这里通常是标量动作） ----------
        act = np.squeeze(actions)
        if act.ndim == 0:
            act = np.full(T, float(act))
        elif act.ndim == 1:
            act = act[:T]
        elif act.ndim == 2:
            # (T,1) -> (T,)
            if act.shape[1] == 1:
                act = act[:, 0]
            # (T, act_dim) -> 先默认画第0维（你如果是标量动作一般不会到这里）
            else:
                act = act[:, 0]
        else:
            act = act.reshape(T, -1)[:, 0]

        # clip 以免数值越界导致图不好看
        act = np.clip(act, -1.0, 1.0)

        # ---------- 画图：3x2 ----------
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1) 总资产曲线
        ax = axes[0, 0]
        ax.plot(t, total_asset, "b-", linewidth=2, label="Total Asset")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Asset")
        ax.set_title(f"{self.name} - Total Asset Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        initial_value = float(total_asset[0])
        final_value = float(total_asset[-1])
        total_ret = (final_value / initial_value - 1.0) * 100.0 if abs(initial_value) > eps else 0.0
        ax.text(
            0.05, 0.95,
            f"Initial: {initial_value:.2f}\nFinal: {final_value:.2f}\nReturn: {total_ret:.2f}%",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        # 2) 持仓占总资产百分比
        ax = axes[0, 1]
        ax.plot(t, holding_pct, "m-", linewidth=2, label="Holding / Total Asset (%)")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Percent (%)")
        ax.set_title(f"{self.name} - Holding Percentage")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.text(
            0.05, 0.95,
            f"Mean: {np.mean(holding_pct):.2f}%\nMax: {np.max(holding_pct):.2f}%",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.5)
        )

        # 3) 持仓股数（可叠加价格用双轴，下面给你默认只画股数）
        ax = axes[1, 0]
        if shares.ndim == 1:
            ax.plot(t, shares, "g-", linewidth=2, label="Shares")
        else:
            # 多标的：每列一条
            for j in range(shares.shape[1]):
                ax.plot(t, shares[:, j], linewidth=1.6, label=f"Shares[{j}]")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Shares")
        ax.set_title(f"{self.name} - Portfolio Shares")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

        # 4) Action plot（每一步买卖百分比，[-1,1]）
        ax = axes[1, 1]
        colors = np.where(act >= 0, "tab:green", "tab:red")
        ax.bar(t, act, color=colors, alpha=0.7, width=1.0, label="Action (buy/sell %)")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action")
        ax.set_title(f"{self.name} - Action Per Step")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 5) 每步策略收益（百分比）
        ax = axes[2, 0]
        ax.plot(t, returns, "r-", linewidth=1.5, alpha=0.8, label="Strategy Return (%)")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Return (%)")
        ax.set_title(f"{self.name} - Strategy Return Time Series")
        ax.grid(True, alpha=0.3)
        ax.legend()

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret / std_ret) * np.sqrt(365*64) if std_ret > 1e-8 else 0.0
        ax.text(
            0.05, 0.95,
            f"Mean: {mean_ret:.4f}%\nStd: {std_ret:.4f}%\nSharpe(ann): {sharpe:.2f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5)
        )

        # 6) 回撤曲线（基于 total_asset）
        ax = axes[2, 1]
        peak = np.maximum.accumulate(total_asset)
        drawdown = np.where(peak > eps, (peak - total_asset) / peak * 100.0, 0.0)
        ax.fill_between(t, drawdown, 0, alpha=0.3, color="red", label="Drawdown (%)")
        ax.plot(t, drawdown, "r-", linewidth=1.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"{self.name} - Drawdown Curve")
        ax.grid(True, alpha=0.3)
        ax.legend()

        max_dd = float(np.max(drawdown)) if drawdown.size else 0.0
        max_dd_idx = int(np.argmax(drawdown)) if drawdown.size else 0
        ax.text(
            0.05, 0.95,
            f"Max Drawdown: {max_dd:.2f}%",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5)
        )
        if drawdown.size:
            ax.plot(max_dd_idx, drawdown[max_dd_idx], "ro", markersize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Current epoch plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

        # 如你希望函数直接返回，也可以用这个返回值
        return holding_pct