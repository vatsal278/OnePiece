import os
import numpy as np

class Trainer:
    def __init__(self, env, agent, num_episodes, checkpoint_interval, checkpoint_dir, memory_file):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.memory_file = memory_file
        self.agent.load_memory_from_disk(self.memory_file)  # Load memory at the start

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = self.agent.observe(state)
            total_reward = 0

            for t in range(len(self.env.data)):
                action, log_prob = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.agent.observe(next_state)
                self.agent.store_transition(state, action, log_prob, reward, next_state, done)
                self.agent.update_position(action)
                state = next_state
                total_reward += reward

            avg_loss = self.agent.learn()
            self.agent.reward_history.append(total_reward)
            self.agent.log_metrics(episode, total_reward, avg_loss)

            if episode % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}.pth")
                self.agent.save_checkpoint(checkpoint_path)

        print("Training finished.")
