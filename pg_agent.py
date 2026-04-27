import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RunningMeanStd:
    """Mantém média e variância correntes para normalização consistente."""
    def __init__(self, shape, epsilon=1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]
        total       = self.count + batch_count
        delta       = batch_mean - self.mean
        self.mean   = self.mean + delta * batch_count / total
        self.var    = (
            self.var  * self.count +
            batch_var * batch_count +
            delta**2  * self.count * batch_count / total
        ) / total
        self.count  = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean)
                / (np.sqrt(self.var) + 1e-8)).astype(np.float32)


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1     = nn.Linear(state_size, 128)
        self.fc2     = nn.Linear(128, 128)
        self.mean    = nn.Linear(128, action_size)
        self.log_std = nn.Parameter(torch.ones(action_size) * -0.5)

    def forward(self, x):
        x       = torch.relu(self.fc1(x))
        x       = torch.relu(self.fc2(x))
        mean    = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        std     = torch.exp(log_std)
        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.0003, gamma=0.99, clip_epsilon=0.2, update_epochs=8, batch_size=128, entropy_coef=0.05):
        self.state_size           = state_size
        self.action_size          = action_size
        self.gamma                = gamma
        self.clip_epsilon         = clip_epsilon
        self.update_epochs        = update_epochs
        self.batch_size           = batch_size
        self.initial_entropy_coef = entropy_coef
        self.min_entropy_coef     = 0.01
        self.entropy_coef         = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_network  = ActorNetwork(state_size, action_size).to(self.device)
        self.critic_network = CriticNetwork(state_size).to(self.device)
        self.optimizer_actor  = optim.Adam(self.actor_network.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=lr)

        self.action_low  = None
        self.action_high = None

        # FIX #1 — running mean/std global em vez de batch z-score
        self.state_rms = RunningMeanStd(shape=(state_size,))

        self.memory = []

    def update_hyperparameters(self, episode, total_episodes):
        if total_episodes > 0:
            progress = episode / total_episodes
            self.entropy_coef = self.min_entropy_coef + (self.initial_entropy_coef - self.min_entropy_coef) * (1 - progress * 0.7)

    def remember(self, state, raw_action, action, log_prob, reward, next_state, done):
        # FIX #2 — guarda raw_action (pré-clamp) para log_prob consistente no update
        self.memory.append((state, raw_action, action, log_prob, reward, next_state, done))

    def normalize_state(self, state: np.ndarray, update: bool = False) -> np.ndarray:
        state = np.array(state, dtype=np.float64)
        if update:
            self.state_rms.update(state.reshape(-1, self.state_size))
        return self.state_rms.normalize(state).astype(np.float32)

    def act(self, state):
        # FIX #1 — normaliza com running stats, actualiza durante rollout
        state_norm   = self.normalize_state(state, update=True)
        state_tensor = torch.FloatTensor(state_norm).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # FIX #3 — sem graph computacional durante rollout
        with torch.no_grad():
            mean, std  = self.actor_network(state_tensor)
            dist       = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            if hasattr(self, "action_low") and self.action_low is not None:
                action = torch.clamp(raw_action, self.action_low, self.action_high)
            else:
                action = torch.clamp(raw_action, -1, 1)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)  # log_prob do raw

        return (
            action.cpu().numpy().squeeze(),
            raw_action.cpu().numpy().squeeze(),  # FIX #2 — raw para memória
            log_prob.cpu().numpy().squeeze(),
        )

    def compute_returns_and_advantages(self, rewards, dones, values, next_values):
        if isinstance(next_values, torch.Tensor):
            num_envs = next_values.shape[1] if next_values.dim() > 1 else 1
        else:
            num_envs = len(next_values[-1]) if hasattr(next_values[-1], "__len__") else 1

        gae_lambda = 0.95
        advantages  = torch.zeros((len(rewards), num_envs), device=self.device)

        for env_idx in range(num_envs):
            last_gae = 0
            for t in reversed(range(len(rewards))):
                # FIX #4 — float() explícito evita contaminação numpy→tensor
                reward = float(rewards[t][env_idx] if isinstance(rewards[t], (list, np.ndarray, torch.Tensor)) else rewards[t])
                done   = float(dones[t][env_idx]   if isinstance(dones[t],   (list, np.ndarray, torch.Tensor)) else dones[t])

                current_val = values[t][env_idx].item() if values.dim() > 1 else values[t].item()

                if t == len(rewards) - 1:
                    next_val = next_values[-1][env_idx].item() if next_values.dim() > 1 else next_values[-1].item()
                else:
                    next_val = values[t+1][env_idx].item() if values.dim() > 1 else values[t+1].item()

                delta    = reward + self.gamma * next_val * (1 - done) - current_val
                last_gae = float(delta + self.gamma * gae_lambda * (1 - done) * last_gae)
                advantages[t, env_idx] = last_gae

        values_reshaped = values.view(len(rewards), num_envs)
        returns    = advantages + values_reshaped.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        states, raw_actions, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)

        states        = np.array(states,        dtype=np.float32)
        raw_actions   = np.array(raw_actions,   dtype=np.float32)
        old_log_probs = np.array(old_log_probs, dtype=np.float32)
        rewards       = np.array(rewards,       dtype=np.float32)
        dones         = np.array(dones,         dtype=np.float32)
        next_states   = np.array(next_states,   dtype=np.float32)

        if rewards.ndim == 1:
            rewards = rewards[:, None]
            dones   = dones[:,   None]

        num_steps, num_envs = rewards.shape

        states      = states.reshape(-1, self.state_size)
        raw_actions = raw_actions.reshape(-1, self.action_size)
        old_log_probs = old_log_probs.reshape(-1, 1)
        rewards     = rewards.reshape(num_steps, num_envs)
        dones       = dones.reshape(num_steps, num_envs)
        next_states = next_states.reshape(-1, self.state_size)

        states      = self.normalize_state(states,      update=False)
        next_states = self.normalize_state(next_states, update=False)

        states      = torch.FloatTensor(states).to(self.device)
        raw_actions = torch.FloatTensor(raw_actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        values      = self.critic_network(states).view(num_steps, num_envs)
        next_values = self.critic_network(next_states).view(num_steps, num_envs)

        returns, advantages = self.compute_returns_and_advantages(
            rewards, dones, values, next_values
        )

        returns    = returns.view(-1, 1)
        advantages = advantages.view(-1, 1)
        dataset_size = states.shape[0]

        for _ in range(self.update_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx       = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states      = states[batch_indices]
                batch_raw_actions = raw_actions[batch_indices]  # FIX #2 — usa raw
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages  = advantages[batch_indices]
                batch_returns     = returns[batch_indices]

                mean, std     = self.actor_network(batch_states)
                dist          = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_raw_actions).sum(dim=-1, keepdim=True)

                ratio   = torch.exp(new_log_probs - batch_old_log_probs)
                entropy = dist.entropy().mean()

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                actor_loss  = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                values_pred = self.critic_network(batch_states)
                critic_loss = (batch_returns - values_pred).pow(2).mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

        self.memory = []
