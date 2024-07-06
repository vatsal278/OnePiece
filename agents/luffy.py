import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

class PPOAgent:
    def __init__(self, agent_id, observation_space, action_space, model, lr=0.001, gamma=0.90, lam=0.90, eps_clip=0.2, K_epochs=4, buffer_capacity=10000, batch_size=64):
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        self.current_position = None
        self.reward_history = []
        self.loss_history = []
        self.memory_counter = 0

    def observe(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        return state_tensor

    def act(self, observation, epsilon=0.1):
        valid_actions = self.get_valid_actions()

        if np.random.rand() < epsilon:
            return np.random.choice(valid_actions), torch.tensor([0.0])

        with torch.no_grad():
            logits = self.model(observation)
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

        action = action.item()

        if action not in valid_actions:
            action = np.random.choice(valid_actions)
            action_log_prob = torch.tensor([0.0])

        return action, action_log_prob
    
    def get_valid_actions(self):
        if self.current_position is None:
            return [0, 1, 2]
        else:
            return [3, 2]

    def update_position(self, action):
        if action == 0:
            self.current_position = 'LONG'
        elif action == 1:
            self.current_position = 'SHORT'
        elif action == 3:
            self.current_position = None

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
        self.memory['dones'].append(done)
        self.memory_counter += 1

        if self.memory_counter >= self.buffer_capacity:
            self.save_memory_to_disk("memory.pt")
            self.clear_memory()

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        num_steps = len(rewards)
        
        values = values.squeeze()  # Removing the singleton dimension
        next_values = next_values.squeeze()  # Removing the singleton dimension

        for step in reversed(range(num_steps)):
            delta = rewards[step] + self.gamma * next_values[step % next_values.size(0)] * (1 - dones[step]) - values[step % values.size(0)]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
                    
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)  # Convert advantages to tensor

    def learn(self):
        # Collect and prepare memory data
        if len(self.memory['states']) == 0:
            return 0  # No data to learn from

        states = torch.cat(self.memory['states']).to(self.device)
        actions = torch.tensor(self.memory['actions']).to(self.device)
        rewards = torch.tensor(self.memory['rewards']).to(self.device)
        next_states = torch.cat(self.memory['next_states']).to(self.device)
        dones = torch.tensor(self.memory['dones'], dtype=torch.float32).to(self.device)

        values = self.model(states)
        next_values = self.model(next_states)
        old_log_probs = torch.cat(self.memory['log_probs']).to(self.device)
        advantages = self.compute_advantages(rewards, values, next_values, dones)

        losses = []
        for _ in range(self.K_epochs):
            for batch in self.sample_batches():
                states_batch, actions_batch, log_probs_batch, advantages_batch = batch
                logits = self.model(states_batch)
                action_probs = torch.softmax(logits, dim=-1)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(actions_batch)
                ratios = torch.exp(log_probs - log_probs_batch)
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                loss = -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        self.loss_history.append(avg_loss)

        # Clear memory after learning
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        return avg_loss

    def log_metrics(self, episode, total_reward, avg_loss):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Episode {episode} - Total Reward: {total_reward:.2f}, Average Loss: {avg_loss:.5f}, Learning Rate: {current_lr:.5f}")

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")

    def save_memory_to_disk(self, filepath):
        checkpoint = {}
        if self.memory['states']:
            checkpoint['states'] = torch.cat(self.memory['states'])
        if self.memory['actions']:
            checkpoint['actions'] = torch.tensor(self.memory['actions'])
        if self.memory['log_probs']:
            checkpoint['log_probs'] = torch.cat(self.memory['log_probs'])
        if self.memory['rewards']:
            checkpoint['rewards'] = torch.tensor(self.memory['rewards'])
        if self.memory['next_states']:
            checkpoint['next_states'] = torch.cat(self.memory['next_states'])
        if self.memory['dones']:
            checkpoint['dones'] = torch.tensor(self.memory['dones'])
        torch.save(checkpoint, filepath)
        print(f"Memory saved to {filepath}")

    def load_memory_from_disk(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            if 'states' in checkpoint:
                self.memory['states'] = list(checkpoint['states'])
            if 'actions' in checkpoint:
                self.memory['actions'] = list(checkpoint['actions'].numpy())
            if 'log_probs' in checkpoint:
                self.memory['log_probs'] = list(checkpoint['log_probs'])
            if 'rewards' in checkpoint:
                self.memory['rewards'] = list(checkpoint['rewards'].numpy())
            if 'next_states' in checkpoint:
                self.memory['next_states'] = list(checkpoint['next_states'])
            if 'dones' in checkpoint:
                self.memory['dones'] = list(checkpoint['dones'].numpy())
            print(f"Memory loaded from {filepath}")
        else:
            print(f"No memory file found at {filepath}")

    def clear_memory(self):
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        self.memory_counter = 0

    def sample_batches(self):
        num_samples = len(self.memory['states'])
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            states_batch = torch.cat([self.memory['states'][i] for i in batch_indices]).to(self.device)
            actions_batch = torch.tensor([self.memory['actions'][i] for i in batch_indices]).to(self.device)
            log_probs_batch = torch.cat([self.memory['log_probs'][i] for i in batch_indices]).to(self.device)
            advantages_batch = torch.tensor([self.memory['rewards'][i] for i in batch_indices]).to(self.device)
            yield states_batch, actions_batch, log_probs_batch, advantages_batch
