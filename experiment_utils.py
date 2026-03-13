import json
import numpy as np
import torch as th
from datetime import datetime
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import mlflow
import os

from MEC_Environment.dual_timescale_wrapper import MECDualTimeScaleEnv, FastEnvWrapper, SlowEnvWrapper
from Optimizer.Random_Dual_Optimizer import RandomDualOptimizer
from Optimizer.Popularity_Optimizer import PopularityOptimizer
from Optimizer.Caching_Gain_Optimizer import CachingGainOptimizer
from Optimizer.Offload_Schemes import LocalOffloadOptimizer
from Optimizer.Exit_Schemes import NoExitOptimizer


def load_env_args():
    """Loads environment arguments from a JSON file."""
    cfg_path = "env_args.json"
    with open(cfg_path, "r") as f:
        return json.load(f)


def load_changing_params():
    """Loads changing experiment parameters from a JSON file."""
    cfg_path = os.path.join("Experiment", "changing_params.json")
    if not os.path.exists(cfg_path):
        # Fallback if not in Experiment folder (e.g. running from root)
        cfg_path = "changing_params.json"
    
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def set_stage_logger(agent, stage_name, base_log_dir):
    """
    Attach a TensorBoard-only logger with a stage-specific namespace.
    """
    log_path = f"{base_log_dir}/{stage_name}"
    logger = configure(folder=log_path, format_strings=["tensorboard"])
    agent.set_logger(logger)


def train_agents(dual_env, slow_episodes, fast_steps_per_slow_episode, slow_agent_type="RL", seed=42):
    """
    Trains the agent(s).
    If slow_agent_type is 'RL', trains both slow and fast agents.
    If slow_agent_type is a heuristic, trains only the fast agent with the heuristic slow agent.

    Returns:
        tuple: The trained slow_agent and fast_agent.
    """
    slow_env = SlowEnvWrapper(dual_env)
    fast_env = FastEnvWrapper(dual_env)

    # Track time for logging
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    # --- Agent Setup ---
    fast_agent_dir = f"tensor_log/A2C_Fast_Agent_{timestamp}"
    fast_agent = A2C("MultiInputPolicy", fast_env, verbose=1, seed=seed, normalize_advantage=True, n_steps=20)
    fast_agent.set_logger(configure(folder=fast_agent_dir, format_strings=["tensorboard", "csv"]))
    fast_agent._setup_learn(
        total_timesteps=1,
        reset_num_timesteps=False
    )
    if slow_agent_type == "RL":
        slow_agent_dir = f"tensor_log/A2C_Slow_Agent_{timestamp}"
        slow_agent = A2C("MultiInputPolicy", slow_env, verbose=1, seed=seed, normalize_advantage=True, n_steps=20)
        slow_agent.set_logger(configure(folder=slow_agent_dir, format_strings=["tensorboard", "csv"]))
        slow_agent._setup_learn(
            total_timesteps=1,
            reset_num_timesteps=False
        )
        log_dir = f"tensorboard_logdir/a2c_training_{slow_agent_type}_{timestamp}"
    elif slow_agent_type == "Popularity":
        slow_agent = PopularityOptimizer(dual_env)
        log_dir = f"tensorboard_logdir/a2c_popularity_{slow_agent_type}_{timestamp}"
    elif slow_agent_type == "CachingGain":
        slow_agent = CachingGainOptimizer(dual_env)
        log_dir = f"tensorboard_logdir/a2c_caching_{slow_agent_type}_{timestamp}"
    else:
        raise ValueError(f"Unknown slow_agent_type: {slow_agent_type}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--- Starting Dual-Timescale Training (Slow Agent: {slow_agent_type}) ---")
    print(f"Logging to {log_dir}")
    # --- Training Loop ---
    global_fast_step = 0
    slow_obs, _ = slow_env.reset()
    fast_obs, _ = fast_env.reset()
    for slow_episode in tqdm(range(slow_episodes), desc=f"Training (Slow: {slow_agent_type})"):
        # --- Slow agent action ---
        if slow_agent_type == "RL":
            slow_action, _ = slow_agent.predict(slow_obs, deterministic=False)
            reshaped_slow_action = slow_action.reshape(
                (dual_env.base_env.num_edges, dual_env.base_env.num_models)
            )
        else: # Heuristic
            reshaped_slow_action = slow_agent.predict_slow(slow_obs)
        slow_env.step(reshaped_slow_action)
        for _ in range(fast_steps_per_slow_episode):
            with th.no_grad():
                obs_tensor = fast_agent.policy.obs_to_tensor(fast_obs)[0]
                fast_action, values, log_prob = fast_agent.policy.forward(obs_tensor, deterministic=False)
            fast_action = fast_action.detach().squeeze().cpu().numpy()
            next_fast_obs, fast_reward, terminated, truncated, info = fast_env.step(fast_action)
            fast_agent.num_timesteps += 1
            writer.add_scalar("Reward/Fast Agent (Per Step)", fast_reward, global_fast_step)
            global_fast_step += 1
            done = bool(terminated or truncated)
            fast_agent.rollout_buffer.add(fast_obs, fast_action, fast_reward, done, values.detach(), log_prob.detach())
            if fast_agent.rollout_buffer.full:
                fast_agent.rollout_buffer.compute_returns_and_advantage(last_values=th.zeros((1,)), dones=np.array([done], dtype=bool))
                fast_agent.train()
                fast_agent.rollout_buffer.reset()
                fast_agent.logger.dump(step=fast_agent.num_timesteps)
            fast_obs = next_fast_obs
        slow_reward, next_slow_obs, cum_fast_reward = slow_env.get_slow_reward_observations()
        # --- Slow agent update (only for RL) ---
        if slow_agent_type == "RL":
            obs_tensor = slow_agent.policy.obs_to_tensor(slow_obs)[0]
            action_tensor = th.as_tensor(slow_action).to(slow_agent.device)
            values, log_prob, _ = slow_agent.policy.evaluate_actions(obs_tensor, action_tensor.unsqueeze(0))
            slow_agent.rollout_buffer.add(slow_obs, slow_action, slow_reward, False, values.detach(), log_prob.detach())
            if slow_agent.rollout_buffer.full:
                slow_agent.rollout_buffer.compute_returns_and_advantage(last_values=th.zeros((1,), device=slow_agent.device), dones=np.array([False], dtype=bool))
                slow_agent.train()
                slow_agent.rollout_buffer.reset()
            writer.add_scalar("Reward/Slow Agent (Cache Hit Rate)", slow_reward, slow_episode)
            slow_agent.num_timesteps += 1
            slow_agent.logger.dump(step=slow_agent.num_timesteps)
        slow_obs = next_slow_obs
    writer.close()
    print(f"\n--- Training completed for {slow_agent_type} slow agent ✅ ---")
    fast_agent.save(f"checkpoints/fast_agent_a2c_{timestamp}")
    slow_agent.save(f"checkpoints/slow_agent_a2c_{timestamp}")
    return slow_agent, fast_agent


def train_caching_only(
    dual_env,
    slow_episodes,
    fast_steps_per_slow,
    log_dir,
    seed=42,
    device="cpu",
):
    """Pre-trains only the slow timescale (caching) agent with specified fast actions."""
    slow_env = SlowEnvWrapper(dual_env)
    fast_env = FastEnvWrapper(dual_env)
    # Track time for logging
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    slow_agent_dir = f"tensor_log/A2C_Slow_Agent_{timestamp}"
    slow_agent = A2C("MultiInputPolicy", slow_env, verbose=0, seed=seed, n_steps=10, device=device)
    set_stage_logger(slow_agent, "Caching_Pretrain", slow_agent_dir)
    slow_agent._setup_learn(total_timesteps=1, reset_num_timesteps=False)
    #slow_agent.set_logger(configure(folder="tensor_log", format_strings=["stdout"]))
    
    writer = SummaryWriter(log_dir=log_dir)

    print(f"--- Caching Pretraining ({slow_episodes} slow episodes) ---")
    print("    Environment: offload_oblivious=True, exit_oblivious=False ---")
    global_fast_step = 0
    slow_obs, _ = slow_env.reset()
    fast_obs, _ = fast_env.reset()
    # Initialize fast optimizer if needed
    offload_optimizer = LocalOffloadOptimizer(dual_env)
    exit_optimizer = NoExitOptimizer(dual_env)
    for slow_episode in tqdm(range(slow_episodes), desc="Caching Pretrain", ncols=100):
        slow_action, _ = slow_agent.predict(slow_obs, deterministic=False)
        reshaped_slow_action = slow_action.reshape(
            (dual_env.base_env.num_edges, dual_env.base_env.num_models)
        )
        slow_env.step(reshaped_slow_action)
        for _ in range(fast_steps_per_slow):
            # The fast_env will expect only exit actions (offload is fixed by env)
            offload_actions = offload_optimizer.predict(fast_obs)
            exit_actions = exit_optimizer.predict(fast_obs)
            fast_action = np.concatenate([offload_actions, exit_actions]).astype(int)
            next_fast_obs, reward, terminated, truncated, info = fast_env.step(fast_action)
            done = bool(terminated or truncated)
            global_fast_step += 1
            fast_obs = next_fast_obs
            if done:
                fast_obs, _ = fast_env.reset()
        # --- Slow agent update ---
        slow_reward, next_slow_obs, cum_fast_reward = slow_env.get_slow_reward_observations()
        obs_tensor = slow_agent.policy.obs_to_tensor(slow_obs)[0]
        action_tensor = th.as_tensor(slow_action).to(slow_agent.device)
        values, log_prob, _ = slow_agent.policy.evaluate_actions(
            obs_tensor, action_tensor.unsqueeze(0)
        )

        slow_agent.rollout_buffer.add(slow_obs, slow_action, slow_reward, False, values.detach(), log_prob.detach())
        slow_agent.num_timesteps += 1
        if slow_agent.rollout_buffer.full:
            slow_agent.rollout_buffer.compute_returns_and_advantage(
                last_values=th.zeros((1,), device=slow_agent.device),
                dones=np.array([False], dtype=bool),
            )
            slow_agent.train()
            slow_agent.rollout_buffer.reset()
        slow_agent.logger.dump(step=slow_agent.num_timesteps)
        slow_obs = next_slow_obs
        writer.add_scalar("Reward/SlowCaching", slow_reward, slow_episode)

    writer.close()
    print("--- Caching pretraining complete ---\n")
    return slow_agent

def train_fast_only(
    dual_env,
    slow_agent,
    train_episodes,
    fast_steps_per_slow,
    log_dir,
    seed=42,
    device="cpu",
):
    """Trains only the fast timescale (offloading/exit) agent while the slow agent provides caching decisions without learning."""
    slow_env = SlowEnvWrapper(dual_env)
    fast_env = FastEnvWrapper(dual_env)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    
    # Initialize optimizers for oblivious settings
    offload_optimizer = None
    exit_optimizer = None
    
    if dual_env.base_env.offload_oblivious:
        offload_optimizer = LocalOffloadOptimizer(dual_env)
    if dual_env.base_env.exit_oblivious:
        exit_optimizer = NoExitOptimizer(dual_env)
    
    # Initialize fast agent
    fast_agent_dir = f"tensor_log/A2C_Fast_Agent_{timestamp}"
    fast_agent = A2C("MultiInputPolicy", fast_env, verbose=0, seed=seed, n_steps=20, normalize_advantage=True, device=device)
    set_stage_logger(fast_agent, "Fast_Only_Train", fast_agent_dir)
    fast_agent._setup_learn(total_timesteps=1, reset_num_timesteps=False)
    #fast_agent.set_logger(configure(folder=None, format_strings=["stdout"]))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--- Fast-Only Training ({train_episodes} slow episodes) ---")
    print(f"    Offload Oblivious: {dual_env.base_env.offload_oblivious}, Exit Oblivious: {dual_env.base_env.exit_oblivious}")
    global_fast_step = 0
    slow_obs, _ = slow_env.reset()
    fast_obs, _ = fast_env.reset()
    for slow_episode in tqdm(range(train_episodes), desc="Fast-Only Train", ncols=100):
        # Slow agent predicts but does not learn
        if hasattr(slow_agent, "predict"): # RL agent
            slow_action, _ = slow_agent.predict(slow_obs, deterministic=True)
            reshaped_slow_action = slow_action.reshape(
                (dual_env.base_env.num_edges, dual_env.base_env.num_models)
            )
        else: # Heuristic optimizer
            reshaped_slow_action = slow_agent.predict_slow(slow_obs)
        slow_env.step(reshaped_slow_action)
        for _ in range(fast_steps_per_slow):
            with th.no_grad():
                obs_tensor = fast_agent.policy.obs_to_tensor(fast_obs)[0]
                agent_action, values, log_prob = fast_agent.policy.forward(obs_tensor, deterministic=False)
            if offload_optimizer is not None:
                offload_actions = offload_optimizer.predict(fast_obs)
                fast_action = np.concatenate([offload_actions, agent_action.detach().cpu().numpy().squeeze()]).astype(int)
            elif exit_optimizer is not None:
                exit_actions = exit_optimizer.predict(fast_obs)
                fast_action = np.concatenate([agent_action.detach().cpu().numpy().squeeze(), exit_actions]).astype(int)
            else:
                fast_action = agent_action.detach().cpu().numpy().squeeze()
            next_fast_obs, reward, terminated, truncated, info = fast_env.step(fast_action)
            done = bool(terminated or truncated)
            writer.add_scalar("Reward/FastRL_Only", reward, global_fast_step)
            global_fast_step += 1
            fast_agent.num_timesteps += 1
            
            # Only update rollout buffer if we're training the fast agent (not fully oblivious)
            fast_agent.rollout_buffer.add(
                fast_obs, agent_action, reward, done, values.detach(), log_prob.detach()
            )
            if fast_agent.rollout_buffer.full:
                with th.no_grad():
                    last_values = fast_agent.policy.predict_values(
                        fast_agent.policy.obs_to_tensor(next_fast_obs)[0]
                    )
                fast_agent.rollout_buffer.compute_returns_and_advantage(
                    last_values=last_values, dones=np.array([done], dtype=bool)
                )
                fast_agent.train()
                fast_agent.rollout_buffer.reset()
            fast_agent.logger.dump(step=fast_agent.num_timesteps)
            fast_obs = next_fast_obs
            if done:
                fast_obs, _ = fast_env.reset()
        # Get slow reward/obs but don't update slow agent
        slow_reward, next_slow_obs, cum_fast_reward = slow_env.get_slow_reward_observations()
        slow_obs = next_slow_obs
        writer.add_scalar("Reward/SlowCaching_Fixed", slow_reward, slow_episode)

    writer.close()
    print("--- Fast-only training complete ---\n")
    return fast_agent

def train_joint_agents(
    dual_env,
    slow_agent,
    fast_agent,
    slow_episodes,
    fast_steps_per_slow,
    log_dir,
    seed=42
):
    """Trains fast offloading/exit agents while continuing to improve the slow agent."""
    slow_env = SlowEnvWrapper(dual_env)
    fast_env = FastEnvWrapper(dual_env)
    
    # Initialize optimizers for oblivious settings
    offload_optimizer = None
    exit_optimizer = None
    
    if dual_env.base_env.offload_oblivious:
        offload_optimizer = LocalOffloadOptimizer(dual_env)
    if dual_env.base_env.exit_oblivious:
        exit_optimizer = NoExitOptimizer(dual_env)
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    fast_agent_dir = f"tensor_log/A2C_Fast_Agent_{timestamp}"
    set_stage_logger(fast_agent, "Joint_Train", fast_agent_dir)
    fast_agent._setup_learn(total_timesteps=1, reset_num_timesteps=False)
    slow_agent_dir = f"tensor_log/A2C_Slow_Agent_{timestamp}"
    set_stage_logger(slow_agent, "Joint_Train", slow_agent_dir)
    slow_agent._setup_learn(total_timesteps=1, reset_num_timesteps=False)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--- Joint training ({slow_episodes} slow episodes) ---")
    print(f"    Offload Oblivious: {dual_env.base_env.offload_oblivious}, Exit Oblivious: {dual_env.base_env.exit_oblivious}")
    global_fast_step = 0
    slow_obs, _ = slow_env.reset()
    fast_obs, _ = fast_env.reset()
    for slow_episode in tqdm(range(slow_episodes), desc="Joint Training", ncols=100):
        slow_action, _ = slow_agent.predict(slow_obs, deterministic=False)
        reshaped_slow_action = slow_action.reshape(
            (dual_env.base_env.num_edges, dual_env.base_env.num_models)
        )
        slow_env.step(reshaped_slow_action)
        for _ in range(fast_steps_per_slow):
            with th.no_grad():
                obs_tensor = fast_agent.policy.obs_to_tensor(fast_obs)[0]
                agent_action, values, log_prob = fast_agent.policy.forward(obs_tensor, deterministic=False)
            
            # Handle oblivious cases for fast actions
            if offload_optimizer is not None:
                offload_actions = offload_optimizer.predict(fast_obs)
                fast_action = np.concatenate([offload_actions, agent_action.detach().cpu().numpy().squeeze()]).astype(int)
            elif exit_optimizer is not None:
                exit_actions = exit_optimizer.predict(fast_obs)
                fast_action = np.concatenate([agent_action.detach().cpu().numpy().squeeze(), exit_actions]).astype(int)
            else:
                fast_action = agent_action.detach().cpu().numpy().squeeze()
            
            next_fast_obs, reward, terminated, truncated, info = fast_env.step(fast_action)
            done = bool(terminated or truncated)
            writer.add_scalar("Reward/FastRL", reward, global_fast_step)
            global_fast_step += 1
            fast_agent.num_timesteps += 1
            fast_agent.rollout_buffer.add(
                fast_obs, agent_action, reward, done, values.detach(), log_prob.detach()
            )
            if fast_agent.rollout_buffer.full:
                with th.no_grad():
                    last_values = fast_agent.policy.predict_values(
                        fast_agent.policy.obs_to_tensor(next_fast_obs)[0]
                    )
                fast_agent.rollout_buffer.compute_returns_and_advantage(
                    last_values=last_values, dones=np.array([done], dtype=bool)
                )
                fast_agent.train()
                fast_agent.rollout_buffer.reset()
            fast_obs = next_fast_obs
            fast_agent.logger.dump(step=fast_agent.num_timesteps)
            if done:
                fast_obs, _ = fast_env.reset()
        # --- Update slow agent ---
        # total_requests = sum(edge.total_requests for edge in dual_env.base_env.edge_servers)
        # cache_hits = sum(edge.cache_hits for edge in dual_env.base_env.edge_servers)
        # slow_reward = cache_hits / total_requests if total_requests > 0 else 0.0
        slow_reward, next_slow_obs, cum_fast_reward = slow_env.get_slow_reward_observations()
        slow_agent.num_timesteps += 1
        for edge in dual_env.base_env.edge_servers:
            edge.cache_hits = 0
            edge.total_requests = 0
        obs_tensor = slow_agent.policy.obs_to_tensor(slow_obs)[0]
        action_tensor = th.as_tensor(slow_action).to(slow_agent.device)
        values, log_prob, _ = slow_agent.policy.evaluate_actions(
            obs_tensor, action_tensor.unsqueeze(0)
        )
        slow_agent.rollout_buffer.add(slow_obs, slow_action, slow_reward, False, values.detach(), log_prob.detach())
        if slow_agent.rollout_buffer.full:
            slow_agent.rollout_buffer.compute_returns_and_advantage(
                last_values=th.zeros((1,), device=slow_agent.device),
                dones=np.array([False], dtype=bool),
            )
            slow_agent.train()
            slow_agent.rollout_buffer.reset()
        slow_obs = next_slow_obs
        slow_agent.logger.dump(step=slow_agent.num_timesteps)
        writer.add_scalar("Reward/SlowCaching", slow_reward, slow_episode)
    # Save the fast agent and small agent
    fast_agent.save(f"checkpoints/fast_agent_a2c_{timestamp}")
    slow_agent.save(f"checkpoints/slow_agent_a2c_{timestamp}")
    writer.close()
    print("--- Joint training complete ---\n")
    return slow_agent, fast_agent

def evaluate_agents(dual_env, slow_agent, fast_agent, num_slow_episodes, fast_steps_per_slow, slow_agent_type="RL", fast_agent_type="RL", seed=42):
    """Evaluates the trained agents against a baseline."""
    all_rewards, all_delays, all_accuracy, all_cache_hit_rates = [], [], [], []
    all_waitings, all_processing, all_transmissions = [], [], []

    slow_env = SlowEnvWrapper(dual_env)
    fast_env = FastEnvWrapper(dual_env)

    desc = f"Evaluating {slow_agent_type}/{fast_agent_type}"
    fast_obs, _ = fast_env.reset()
    slow_obs, _ = slow_env.reset()
    for _ in tqdm(range(num_slow_episodes), desc=desc, ncols=100):
        # --- Slow Agent Action ---
        if slow_agent_type == "RL":
            slow_action, _ = slow_agent.predict(slow_obs, deterministic=True)
            reshaped_slow_action = np.round(slow_action).astype(int).reshape(
                (dual_env.base_env.num_edges, dual_env.base_env.num_models)
            )
        else:  # Heuristic (Random, Popularity, CachingGain)
            reshaped_slow_action = slow_agent.predict_slow(slow_obs)
        dual_env.step_slow(reshaped_slow_action)
        for _ in range(fast_steps_per_slow):
            # --- Fast Agent Action ---
            if fast_agent_type == "RL":
                fast_action, _ = fast_agent.predict(fast_obs, deterministic=True)
            else:  # Heuristic (Random fast part)
                fast_action = fast_agent.predict_fast(fast_obs)
            fast_obs, reward, terminated, truncated, info = fast_env.step(fast_action)
            if info.get("num_tasks", 0) > 0:
                all_rewards.append(reward)
                all_delays.append(info["latency"])
                all_accuracy.append(info["accuracy"])
                all_waitings.append(info["waiting_time"])
                all_processing.append(info["compute_latency"])
                all_transmissions.append(info["transmit_latency"])
            # if terminated or truncated:
            #     fast_obs, _ = fast_env.reset()

        # Calculate cache hit rate for the slow episode
        slow_reward, slow_obs, cum_fast_reward = slow_env.get_slow_reward_observations()
        # total_requests = sum(edge.total_requests for edge in dual_env.base_env.edge_servers)
        # cache_hits = sum(edge.cache_hits for edge in dual_env.base_env.edge_servers)
        # all_cache_hit_rates.append(cache_hits / total_requests if total_requests > 0 else 0.0)
        all_cache_hit_rates.append(slow_reward)
        
        for edge in dual_env.base_env.edge_servers:
            edge.cache_hits = 0
            edge.total_requests = 0
            
    metrics = {
        "reward": np.mean(all_rewards),
        "delay": np.mean(all_delays),
        "accuracy": np.mean(all_accuracy),
        "cache_hit_rate": np.mean(all_cache_hit_rates),
        "waiting_time": np.mean(all_waitings),
        "processing_time": np.mean(all_processing),
        "transmission_time": np.mean(all_transmissions),
    }
    return metrics


def save_mlflow_dual(optimizer_name, experiment_name, setting, env_params, metrics, db_name="mlflow.db", log_dir="mlruns_db"):
    """Logs experiment results to MLflow, ensuring database tracking and artifact location."""
    # Set tracking URI to a local SQLite database
    db_path = os.path.abspath(db_name)
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Ensure artifact directory exists
    artifact_path = os.path.abspath(log_dir)
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
        
    # Ensure experiment exists with correct artifact location
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, 
            artifact_location=f"file:///{artifact_path}"
        )
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{optimizer_name}_{setting}_Dual"):
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("setting", setting)
        mlflow.log_params(env_params)
        mlflow.log_metrics(metrics)