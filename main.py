import yaml
import pandas as pd
from environments.base_env import CryptoTradingEnv
from models.lstm_attention import StackedLSTMWithAttention
from agents.luffy import PPOAgent
from trainers.trainer import Trainer

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_config('config/config.yaml')

    data = pd.read_csv(config['env']['data_file'])
    data = data.drop(columns=['Open Time', 'Close Time', 'Ignore'])
    env = CryptoTradingEnv(data, 
                           initial_balance=config['env']['initial_balance'], 
                           commission=config['env']['commission'], 
                           slippage=config['env']['slippage'])

    model = StackedLSTMWithAttention(input_dim=env.observation_space.shape[0], 
                                     hidden_dim=64, 
                                     output_dim=env.action_space.n)

    agent = PPOAgent(agent_id=1, 
                     observation_space=env.observation_space, 
                     action_space=env.action_space, 
                     model=model)

    trainer = Trainer(env, agent, 
                      num_episodes=config['trainer']['num_episodes'], 
                      checkpoint_interval=config['trainer']['checkpoint_interval'], 
                      checkpoint_dir=config['trainer']['checkpoint_dir'])

    trainer.train()
