import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, agent_id, observation_space, action_space, model, lr=0.001, gamma=0.90, lam=0.90, eps_clip=0.2, K_epochs=4):
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

        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        self.current_position = None

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
            logits = self.model(states)
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update old_log_probs for the next iteration
            old_log_probs = log_probs.detach()

            losses.append(loss.item())
        
        avg_loss = np.mean(losses)

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