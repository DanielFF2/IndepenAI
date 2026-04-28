IndepenAI
----------------------------------------------------------------------------------------------------------------------------------
IndepenAI is a modular reinforcement learning agent based on Proximal Policy Optimization (PPO), implemented in PyTorch. It is designed to work with OpenAI Gym / Gymnasium environments and supports continuous action spaces. IndepenAI is ideal for research, experimentation, and learning about modern RL techniques.

----------------------------------------------------------------------------------------------------------------------------------

## Architecture

IndepenAI is built around three main components:
ActorNetwork — Decides which action to take given the current state. Outputs the mean and standard deviation of a Gaussian distribution for continuous action spaces.

CriticNetwork — Estimates the value of a given state, helping the agent understand how good or bad a situation is.

PPOAgent — Manages the full learning pipeline: collecting experiences, computing GAE advantages, normalising states via a running mean/std (RunningMeanStd), and updating both networks using PPO's clipped objective.

Key implementation details in the current version:

Running state normalisation — RunningMeanStd keeps a global running mean/variance instead of per-batch z-score, preventing distribution drift across episodes.
Raw action storage — raw_action (pre-clamp) is stored in memory and used for log-probability computation during updates, keeping the policy gradient consistent.
No-gradient rollouts — torch.no_grad() is used during environment interaction to avoid graph contamination.
GAE-λ advantages — Generalised Advantage Estimation with λ = 0.95 for variance reduction.
Gradient clipping — max_norm = 0.5 on both actor and critic to stabilise training.
Plateau detection — Entropy coefficient is dynamically adjusted when recent reward improvement stalls.

----------------------------------------------------------------------------------------------------------------------------------

## What Can Be Customised

Network sizes — Change the number of layers or neurons in ActorNetwork and CriticNetwork.
PPO hyperparameters — Learning rate, gamma, clip epsilon, batch size, entropy coefficient, update epochs.
Environments — Any Gym-compatible environment with a continuous action space.
Checkpointing — Save and resume training at any point; the checkpoint includes normalisation statistics, optimizer states, and episode rewards.

----------------------------------------------------------------------------------------------------------------------------------

## Installation

Python 3.9 is recommended for best compatibility with Gym and Box2D environments.
On Windows, make sure Python is added to your PATH in environment variables.
Install Dependencies
    
    bashpip install --upgrade pip wheel setuptools
    pip install torch numpy matplotlib gym[box2d]
    # [box2d] is required for BipedalWalker environments
    # Add gym[all] or other extras for additional environments
    
If you encounter wheel build errors (Box2D, MuJoCo), try pinning the gym version:

    bashpip install gym==0.21.0
or

    pip install box2d-py
Clone the Repository

    git clone https://github.com/DanielFF2/IndepenAI.git
    cd IndepenAI

----------------------------------------------------------------------------------------------------------------------------------

Example — BipedalWalker Hardcore (PPO)

This is the full training script used with the current pg_agent.py. It uses 16 parallel environments, running state normalisation, and saves checkpoints every 50 episodes.

    from pg_agent import PPOAgent
    import numpy as np
    
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    
    import torch
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import gym
    import os
    
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    # --- Setup ---
    
    num_envs      = 16
    num_episodes  = 1000
    max_timesteps = 4000
    
    env = gym.vector.make("BipedalWalkerHardcore-v3", num_envs=num_envs, asynchronous=False)
    state_size  = env.single_observation_space.shape[0]
    action_size = env.single_action_space.shape[0]
    print(f"state_size: {state_size} | action_size: {action_size}")
    
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        update_epochs=6,
        batch_size=128,
        entropy_coef=0.02,
    )
    
    agent.action_low  = torch.tensor(env.single_action_space.low,  device=agent.device)
    agent.action_high = torch.tensor(env.single_action_space.high, device=agent.device)
    
    # --- Checkpoint loading ---
    
    checkpoint_path  = "bipedalwalkerhardcore_ppo_checkpoint.pt"
    episode_rewards  = []
    starting_episode = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.actor_network.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_network.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        agent.state_rms.mean  = checkpoint['state_rms_mean']
        agent.state_rms.var   = checkpoint['state_rms_var']
        agent.state_rms.count = checkpoint['state_rms_count']
        agent.entropy_coef    = checkpoint.get('entropy_coef', 0.02)
        starting_episode      = checkpoint['episode'] + 1
        episode_rewards       = checkpoint['rewards']
        print(f"Checkpoint loaded: continuing from episode {starting_episode}")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    # --- Training loop ---
    
    for episode in range(starting_episode, starting_episode + num_episodes):
        state         = env.reset()[0]
        total_rewards = np.zeros(num_envs)
        dones         = np.zeros(num_envs, dtype=bool)

    for t in range(max_timesteps):
        action, raw_action, log_prob = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, raw_action, action, log_prob, reward, next_state, done)
        total_rewards += reward
        state = next_state
        dones = np.logical_or(dones, done)
        if np.all(dones):
            break

    agent.update()
    mean_reward = np.mean(total_rewards)
    episode_rewards.append(mean_reward)
    print(f"Episode {episode + 1} | Mean reward: {mean_reward:.2f}")

    # Plateau detection — adjust entropy coefficient
    episodes_this_run = episode - starting_episode
    if episodes_this_run >= 100:
        recent_mean = np.mean(episode_rewards[-50:])
        older_mean  = np.mean(episode_rewards[-100:-50])
        if abs(recent_mean - older_mean) < 15.0:
            agent.entropy_coef = min(agent.entropy_coef * 1.1, 0.05)
            print(f"  [PLATEAU] Entropy increased to {agent.entropy_coef:.4f}")
        else:
            agent.entropy_coef = max(agent.entropy_coef * 0.99, 0.02)

    # Save checkpoint every 50 episodes
    if (episode + 1) % 50 == 0:
        torch.save({
            'entropy_coef':                agent.entropy_coef,
            'state_rms_mean':              agent.state_rms.mean,
            'state_rms_var':               agent.state_rms.var,
            'state_rms_count':             agent.state_rms.count,
            'actor_state_dict':            agent.actor_network.state_dict(),
            'critic_state_dict':           agent.critic_network.state_dict(),
            'optimizer_actor_state_dict':  agent.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
            'episode':                     episode,
            'rewards':                     episode_rewards,
        }, checkpoint_path)
        print(f"Checkpoint saved at episode {episode + 1}")

    env.close()
    
    # --- Plot training curve ---
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, linewidth=0.8)
    plt.title("Episode Rewards (BipedalWalker Hardcore - PPO)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("Training curve saved.")

----------------------------------------------------------------------------------------------------------------------------------

## Checkpoint Format

The checkpoint includes everything needed to resume training with no loss of state:
actor_state_dict - Actor network weight
critic_state_dict - Critic network weight
optimizer_actor_state_dict - Actor optimizer state
optimizer_critic_state_dict - Critic optimizer state
state_rms_mean / _var / _count - Running normalisation statistics
entropy_coef - Current entropy coefficient
episode - Last completed episode 
indexrewards - Full list of episode rewards

## Using Other Environments

Change the environment name in gym.vector.make() and update state_size / action_size accordingly. 
The agent works with any Gym-compatible environment that has a continuous action space.
For larger or more complex environments, it is recommended to split your code into an environment.py and a main.py.

----------------------------------------------------------------------------------------------------------------------------------
## Contributing

Contributions are welcome! Open an issue or submit a pull request with suggestions, bug fixes, or improvements — hyperparameter changes, new mechanics, anything that makes the agent learn more efficiently is appreciated.

## License

This project is independent (as the name sugests), so no issues with that!
