import gym
from gym import spaces
import pandas as pd
import numpy as np

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1000000, commission=0.003, slippage=0):
        super(CryptoTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.entry_step = 0
        self.trades = []
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward()
        return self._next_observation(), reward, done, {}

    def _calculate_reward(self):
        reward = 0
        if self.position is not None:
            current_price = self.data.iloc[self.current_step]['Close']
            price_diff = current_price - self.entry_price
            if self.position == 'LONG':
                reward = price_diff
            elif self.position == 'SHORT':
                reward = -price_diff

            trade_size = 0.0001 * self.balance
            commission = self.commission * trade_size
            reward -= (commission + self.slippage)
        return reward

    def _take_action(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        if action == 0 and self.position is None:
            self._open_position('LONG')
        elif action == 1 and self.position is None:
            self._open_position('SHORT')
        elif action == 2:
            pass
        elif action == 3 and self.position is not None:
            self._close_position(current_price)

    def _open_position(self, position_type):
        self.position = position_type
        self.entry_price = self.data.iloc[self.current_step]['Close']
        self.entry_step = self.current_step

    def _close_position(self, current_price):
        if self.position is None:
            return
        price_diff = current_price - self.entry_price
        trade_size = 0.0001 * self.balance
        profit = price_diff * trade_size / self.entry_price if self.position == 'LONG' else -price_diff * trade_size / self.entry_price
        commission = self.commission * trade_size
        profit -= (commission + self.slippage)
        self.balance += profit
        self.trades.append({
            'step': self.current_step,
            'type': self.position,
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'profit': profit
        })
        self.position = None

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Entry Price: {self.entry_price}')
        print(f'Trades: {self.trades}')
