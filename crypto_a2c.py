import os
import pandas as pd
from copy import deepcopy
from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, EvalCallback


from finrl.meta.env_cryptocurrency_trading.env_btc_ccxt import BitcoinEnv
from finrl.agents.stablebaselines3.a2c_util import *


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    RESULTS = "results"

    env_train_gym = BitcoinEnv(
        data_cwd=".",
        time_frequency=1,
        mode="train"
    )
    env_train, _  = env_train_gym.get_sb_env()
    env_eval_gym = BitcoinEnv(
        data_cwd=".",
        time_frequency=1,
        mode="trade"
    )
    env_eval, _ = env_eval_gym.get_sb_env()
    env_train2 = deepcopy(env_train)
    
    n_step = 2048
    eval_freq_per_n_step = 4
    model_a2c = A2C(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=1e-4,
        n_steps=n_step,
    )

    tmp_path = RESULTS + f'/a2c-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "log"])
    model_a2c.set_logger(new_logger_a2c)


    #############################################################
    #                           train                           #
    #############################################################
    eval_callback = TradeCallback(
        eval_env=env_eval,
        eval_freq=n_step*eval_freq_per_n_step,
        n_eval_episodes=1,
        log_path=tmp_path,
        deterministic=True,
        name="eval_eval"
    )
    train_callback = TradeCallback(
        eval_env=env_train2,
        eval_freq=n_step*eval_freq_per_n_step,
        n_eval_episodes=1,
        log_path=tmp_path,
        deterministic=True,
        name="eval_train"
    )

    callbacks = CallbackList([eval_callback, train_callback])

    model_a2c.learn(
        total_timesteps=5e5,
        log_interval=eval_freq_per_n_step,
        callback=callbacks,
    )