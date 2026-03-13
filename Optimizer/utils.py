import numpy as np
import mlflow
from torch.utils.tensorboard import SummaryWriter


def save_tensorboard(optimizer_name, setting, env_params, delay):
    writer = SummaryWriter(f"runs/optimizer_experiment/{optimizer_name}_{setting}")
    hparams = {"optimizer": optimizer_name, "setting": setting} | env_params
    metrics =  {"hparam/delay": delay}
    writer.add_hparams(hparams, metrics)
    # Log scalar explicitly
    writer.close()

def save_mlflow(optimizer_name, experiment_name, setting, env_params, metrics):
    mlflow.set_experiment(experiment_name)
    # Start an MLflow run
    with mlflow.start_run(run_name=f"{optimizer_name}_{setting}"):
        # Log parameters
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("setting", setting)
        mlflow.log_params(env_params)
        # Log metrics
        mlflow.log_metrics(metrics)

def evaluate_optimizer(env, optimizer, num_episodes=10, verbose=False):
    """
    Evaluates the optimizer over a number of episodes.
    Returns: average reward across episodes.
    """
    rewards = []
    delays = []
    compute_delays = []
    transmit_delays = []
    accuracy = []
    offload_rate = []
    obs, info = env.reset()
    for step in range(num_episodes):
        action = optimizer.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        delays.append(info["latency"])
        compute_delays.append(info["compute_latency"])
        transmit_delays.append(info["transmit_latency"])
        accuracy.append(info["accuracy"])
        offload_rate.append(info["offloadable_rate"])
        if verbose:
            print(f"Step {step}: Action={action}, Reward={reward}, Info={info}")
        if terminated or truncated:
            obs, info = env.reset()
    return rewards, delays, compute_delays, transmit_delays, accuracy, offload_rate