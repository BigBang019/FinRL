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
            collected_forward_return = []
            
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
                collected_forward_return.append(info['env_forward_return'])

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
            forward_return_array = np.array(collected_forward_return) # (n, )

            self.current_epoch_info['total_asset_array'] = total_asset_array
            self.current_epoch_info['portfolio_array'] = portfolio_array
            self.current_epoch_info['forward_return_array'] = forward_return_array
            
            total_return = (total_asset_array[-1] / total_asset_array[0] - 1) * 100
            self.metrics['total_return'].append(total_return)
            
            sharpe_ratio = np.mean(forward_return_array) / np.std(forward_return_array)
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
        if self.current_epoch_info and 'forward_return_array' in self.current_epoch_info:
            returns = self.current_epoch_info['forward_return_array']
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
    
    def plot_current_epoch(self, save_path=None, figsize=(12, 8)):
        """
        专门绘制当前epoch的详细信息
        
        Args:
            save_path: 保存图片的路径，如果为None则不保存
            figsize: 图片大小，默认(12, 8)
        """
        if not self.current_epoch_info:
            print("Warning: No current epoch info to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 总资产曲线
        if 'total_asset_array' in self.current_epoch_info:
            ax1 = axes[0, 0]
            total_asset = self.current_epoch_info['total_asset_array']
            time_steps = np.arange(len(total_asset))
            ax1.plot(time_steps, total_asset, 'b-', linewidth=2, label='Total Asset')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Total Asset')
            ax1.set_title(f'{self.name} - Total Asset Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 添加初始值和最终值标注
            initial_value = total_asset[0]
            final_value = total_asset[-1]
            return_pct = (final_value / initial_value - 1) * 100
            ax1.text(0.05, 0.95, f'Initial: {initial_value:.2f}\nFinal: {final_value:.2f}\nReturn: {return_pct:.2f}%',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. 持仓变化
        if 'portfolio_array' in self.current_epoch_info:
            ax2 = axes[0, 1]
            portfolio = self.current_epoch_info['portfolio_array']
            time_steps = np.arange(len(portfolio))
            ax2.plot(time_steps, portfolio, 'g-', linewidth=2, label='Portfolio Holdings')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Holdings')
            ax2.set_title(f'{self.name} - Portfolio Holdings')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. 收益率时间序列
        if 'forward_return_array' in self.current_epoch_info:
            ax3 = axes[1, 0]
            returns = self.current_epoch_info['forward_return_array']
            time_steps = np.arange(len(returns))
            ax3.plot(time_steps, returns, 'r-', linewidth=1.5, alpha=0.7, label='Return (%)')
            ax3.axhline(0, color='k', linestyle='--', linewidth=1)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Return (%)')
            ax3.set_title(f'{self.name} - Return Time Series')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 添加统计信息
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 1e-8 else 0
            ax3.text(0.05, 0.95, f'Mean: {mean_return:.4f}%\nStd: {std_return:.4f}%\nSharpe: {sharpe:.2f}',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 4. 回撤曲线
        if 'total_asset_array' in self.current_epoch_info:
            ax4 = axes[1, 1]
            total_asset = self.current_epoch_info['total_asset_array']
            peak = np.maximum.accumulate(total_asset)
            drawdown = (peak - total_asset) / peak * 100
            time_steps = np.arange(len(drawdown))
            ax4.fill_between(time_steps, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
            ax4.plot(time_steps, drawdown, 'r-', linewidth=1.5)
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Drawdown (%)')
            ax4.set_title(f'{self.name} - Drawdown Curve')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 添加最大回撤标注
            max_dd = np.max(drawdown)
            max_dd_idx = np.argmax(drawdown)
            ax4.text(0.05, 0.95, f'Max Drawdown: {max_dd:.2f}%',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            ax4.plot(max_dd_idx, max_dd, 'ro', markersize=10)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Current epoch plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()