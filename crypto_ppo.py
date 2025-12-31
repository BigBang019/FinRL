import os
import pandas as pd
from copy import deepcopy
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, EvalCallback


from finrl.meta.env_cryptocurrency_trading.env_btc_ccxt import *
from finrl.meta.env_cryptocurrency_trading.binance_usdm_env import *
from finrl.agents.stablebaselines3.a2c_util import *


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    RESULTS = "results"

    env_train_gym = BitcoinEnvDSR(
        data_cwd=".",
        time_frequency=1,
        mode="train"
    )
    env_eval_gym = BitcoinEnvDSR(
        data_cwd=".",
        time_frequency=1,
        mode="trade"
    )
    # env_train_gym = USDMEnv(data_cwd=".", mode="train")
    # env_eval_gym = USDMEnv(data_cwd=".", mode='trade')
    env_train, _ = env_train_gym.get_sb_env()
    env_eval, _ = env_eval_gym.get_sb_env()
    env_train2 = deepcopy(env_train)

    
    # PPO关键参数
    n_steps = 2048  # 每次rollout收集的步数
    batch_size = 64  # 每次更新的batch大小（必须 <= n_steps）
    n_epochs = 10  # 对同一批数据重复更新的次数（PPO的关键特性）
    eval_freq_per_n_step = 4  # 评估频率
    
    # PPO模型配置
    model_ppo = PPO(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=3e-4,  # PPO通常使用稍高的学习率
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,  # PPO的核心：多次更新同一批数据
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # GAE lambda参数
        clip_range=0.2,  # PPO的clip范围（关键超参数）
        ent_coef=0.01,  # 熵系数，鼓励探索
        vf_coef=0.5,  # 价值函数损失系数
        max_grad_norm=0.5,  # 梯度裁剪，稳定训练
        tensorboard_log=None,  # 可以设置为tensorboard日志路径
        policy_kwargs=None,  # 可以自定义网络结构
        verbose=1,
        seed=None,
        device="cuda",  # 自动选择CPU或GPU
    )

    tmp_path = RESULTS + f'/ppo-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "log"])
    model_ppo.set_logger(new_logger_ppo)


    #############################################################
    #                           train                           #
    #############################################################
    eval_callback = TradeCallback(
        eval_env=env_eval,
        eval_freq=n_steps*eval_freq_per_n_step,
        n_eval_episodes=5,  # 从1增加到5，提高评估稳定性
        log_path=tmp_path,
        deterministic=True,
        name="eval_eval"
    )
    train_callback = TradeCallback(
        eval_env=env_train2,
        eval_freq=n_steps*eval_freq_per_n_step,
        n_eval_episodes=1,  # 从1增加到5，提高评估稳定性
        log_path=tmp_path,
        deterministic=True,
        name="eval_train"
    )

    callbacks = CallbackList([eval_callback, train_callback])

    model_ppo.learn(
        total_timesteps=5e5,
        log_interval=eval_freq_per_n_step,
        callback=callbacks,
    )

    train_callback.plot(f"{tmp_path}/train_global.png")
    eval_callback.plot(f"{tmp_path}/eval_global.png")