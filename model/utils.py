import torch
import torch.nn as nn
import gym
import os
import scipy.linalg
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yaml

from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Callable
from tqdm import tqdm

from .buffer import DemoBuffer, TrackingBuffer
from .env.base import ControlAffineSystem, TrackingEnv


def set_seed(seed: int, env: gym.Env = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if env is not None:
        env.seed(seed)


def lqr(a: np.array, b: np.array, q: np.array, r: np.array):
    """
    compute the discrete-time LQR controller: a = -k.dot(s)
    dynamics: s_{t+1} = a * s_t + b * a_t
    cost function: \sum (s_t * q * s_t + a_t * r * a_t)

    :param a: np.array
    :param b: np.array
    :param q: np.array
    :param r: np.array
    :return:
    k : np.array
        Controller matrix
    p : np.array
        Cost to go matrix
    """
    p = scipy.linalg.solve_discrete_are(a, b, q, r)
    p = p.astype(np.float32)

    # LQR gain k = (b.T * p * b + r)^-1 * (b.T * p * a)
    bp = b.T.dot(p)
    tmp1 = bp.dot(b)
    tmp1 += r
    tmp2 = bp.dot(a)
    k = np.linalg.solve(tmp1, tmp2)
    return k, p


def reset_env(env: gym.wrappers.time_limit.TimeLimit, init_state: np.array = None, init_range: float = None):
    state = env.reset()
    if init_state is not None:
        env.env.set_state(qpos=init_state[:env.env.model.nq], qvel=init_state[env.env.model.nq:])
        state = env.env._get_obs()
    elif init_range is not None:
        state = np.random.uniform(size=env.env.model.nq + env.env.model.nv, low=-init_range, high=init_range)
        env.env.set_state(qpos=state[:env.env.model.nq], qvel=state[env.env.model.nq:])
        state = env.env._get_obs()
    return state


def eval_policy(
        env: ControlAffineSystem,
        policy,
        device: torch.device,
        n_epi: int = 5,
) -> np.array:
    rewards = []
    lengths = []

    state = env.reset()
    epi_length = 0
    epi_reward = 0
    for i_epi in range(n_epi):
        while True:
            action = policy.act(state)
            next_state, reward, done, _ = env.step(action)
            epi_length += 1
            epi_reward += reward
            state = next_state
            if done:
                rewards.append(epi_reward)
                lengths.append(epi_length)
                epi_reward = 0
                epi_length = 0
                state = env.reset()
                break
    print(f'average reward: {np.mean(rewards):.2f}, average length: {np.mean(lengths):.2f}')
    return np.mean(rewards)


def eval_tracking_policy(
        env: TrackingEnv,
        policy,
        device: torch.device,
        n_epi: int = 5
) -> np.array:
    rewards = []
    lengths = []

    state = env.reset()
    epi_length = 0
    epi_reward = 0
    for i_epi in range(n_epi):
        while True:
            state_ref, action_ref = env.get_ref()
            action = policy.track(state, state_ref, action_ref)
            next_state, reward, done, _ = env.step(action)
            epi_length += 1
            epi_reward += reward
            state = next_state
            if done:
                rewards.append(epi_reward)
                lengths.append(epi_length)
                epi_reward = 0
                epi_length = 0
                state = env.reset()
                break
    print(f'average reward: {np.mean(rewards):.2f}, average length: {np.mean(lengths):.2f}')
    return np.mean(rewards)


def collect_demo(buffer: DemoBuffer, env: ControlAffineSystem, policy, n_steps: int, sample_traj: bool = True):
    epi_rewards = []
    cur_reward = 0
    cur_step = 0

    if sample_traj:
        state = env.reset()
        for i in range(n_steps):
            if policy == 'random':
                action = env.sample_ctrls(1)
            else:
                action = policy.explore(state, std=0.01)
            next_state, reward, done, _ = env.step(action)
            cur_reward += reward
            buffer.append(state, action, reward, done, next_state)
            state = next_state
            cur_step += 1

            if done:
                state = env.reset()
                epi_rewards.append(cur_reward)
                cur_reward = 0
                cur_step = 0
        print(f'mean reward: {buffer.mean_reward:.2f}')
    else:
        states = env.sample_states(n_steps)
        actions = env.sample_ctrls(n_steps)
        next_states = env.forward(states, actions)
        for i in range(states.shape[0]):
            buffer.append(states[i, :], actions[i, :], 0, False, next_states[i, :])
    return buffer


def collect_tracking_demo(buffer: TrackingBuffer, env: TrackingEnv, policy, n_steps: int, device: torch.device):
    epi_rewards = []
    cur_reward = 0
    cur_step = 0

    state = env.reset()
    for i in range(n_steps):
        state_ref, action_ref = env.get_ref()
        action = policy.track(state, state_ref, action_ref)
        next_state, reward, done, _ = env.step(action)
        cur_reward += reward
        buffer.append(state, action, reward, done, next_state, state_ref, action_ref)
        state = next_state
        cur_step += 1

        if done:
            state = env.reset()
            epi_rewards.append(cur_reward)
            cur_reward = 0
            cur_step = 0
    print(f'mean reward: {buffer.mean_reward:.2f}')
    return buffer


def init_logger(
        log_path: str,
        env_name: str,
        algo_name: str,
        seed: int,
        args: dict = None,
        hyper_params: dict = None,
        tensorboard: bool = True,
) -> Tuple[str, SummaryWriter, str]:
    """
    Initialize the logger. The logger dir should include the following path:
        - <log folder>
            - <env name>
                - <algo name>
                    - seed<seed>_<experiment time>
                        - settings.yaml: the experiment setting
                        - summary: training summary
                        - models: saved models

    Parameters
    ----------
    log_path: str
        name of the log folder
    env_name: str
        name of the training environment
    algo_name: str
        name of the algorithm
    seed: int
        random seed used
    args: dict
        arguments to be written down: {argument name: value}
    hyper_params: dict
        hyper-parameters for training
    tensorboard: bool
        whether to use tensorboard

    Returns
    -------
    log_path: str
        path of the log
    writer: SummaryWriter
        summary dir
    model_path: str
        models dir
    """
    # make log path
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # make path with specific env
    if not os.path.exists(os.path.join(log_path, env_name)):
        os.mkdir(os.path.join(log_path, env_name))

    # make path with specific algorithm
    if not os.path.exists(os.path.join(log_path, env_name, algo_name)):
        os.mkdir(os.path.join(log_path, env_name, algo_name))

    # record the experiment time
    start_time = datetime.datetime.now()
    start_time = start_time.strftime('%Y%m%d%H%M%S')
    if not os.path.exists(os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}')):
        os.mkdir(os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}'))

    # set up log, summary writer
    log_path = os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}')
    if tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(log_path, 'summary'))
    else:
        writer = None

    # make path for saving models
    model_path = os.path.join(log_path, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # write args
    log = open(os.path.join(log_path, 'settings.yaml'), 'w')
    if args is not None:
        for key in args.keys():
            log.write(f'{key}: {args[key]}\n')
    if 'algo' not in args.keys():
        log.write(f'algo: {algo_name}\n')
    if hyper_params is not None:
        log.write('hyper_params:\n')
        for key1 in hyper_params.keys():
            if type(hyper_params[key1]) == dict:
                log.write(f'  {key1}: \n')
                for key2 in hyper_params[key1].keys():
                    log.write(f'    {key2}: {hyper_params[key1][key2]}\n')
            else:
                log.write(f'  {key1}: {hyper_params[key1]}\n')
    else:
        log.write('hyper_params: using default hyper-parameters')
    log.close()

    return log_path, writer, model_path


def export_settings(log_path, args: dict):
    log = open(os.path.join(log_path, 'settings.yaml'), 'w')
    if args is not None:
        for key in args.keys():
            log.write(f'{key}: {args[key]}\n')
    log.close()


def plot_demo(
        buffer: DemoBuffer,
        env: ControlAffineSystem,
        initial_iter: int,
        plot_dim: Tuple[int, int],
        path: str,
        demo_iter: int = None,
):
    if demo_iter is None:
        demo_iter = initial_iter

    x_dim, y_dim = plot_dim
    x_all = buffer.all_states[:, x_dim]
    y_all = buffer.all_states[:, y_dim]
    buffer_size = buffer.buffer_size

    assert (buffer_size - initial_iter) % demo_iter == 0

    n_iter = int((buffer_size - initial_iter) / demo_iter)
    x_iter = torch.split(x_all, [initial_iter] + [demo_iter] * n_iter)
    y_iter = torch.split(y_all, [initial_iter] + [demo_iter] * n_iter)

    upper_limit, lower_limit = env.state_limits
    x_limit = (lower_limit[x_dim].cpu(), upper_limit[x_dim].cpu())
    y_limit = (lower_limit[y_dim].cpu(), upper_limit[y_dim].cpu())

    plt.figure()
    for i_iter in tqdm(range(n_iter)):
        plt.scatter(x_iter[i_iter].cpu(), y_iter[i_iter].cpu(), s=1, alpha=0.6, label=f'iter: {i_iter}')
        plt.scatter(env.goal_point[0, x_dim].cpu(), env.goal_point[0, y_dim].cpu(), s=10, alpha=1, c='black')
        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(f'dim: {env.state_name(x_dim)}')
        plt.ylabel(f'dim: {env.state_name(y_dim)}')
        plt.savefig(os.path.join(path, f'demos_x{x_dim}y{y_dim}_{i_iter}.png'))
    plt.close()


def read_settings(path: str):
    with open(os.path.join(path, 'settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings
