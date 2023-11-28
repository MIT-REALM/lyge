import torch
import argparse
import numpy as np
import cv2
import os
import time

from tqdm import tqdm

from model.utils import set_seed, read_settings
from model.controller import NeuralCLFController
from model.env.env import make_env
from model.buffer import DemoBuffer
from model.rl_il.agents import AGENTS, get_agent


def execute_policy(idx, cur_policy, env, args):
    video_path = os.path.join(args.path, 'videos')
    if not args.no_video and not os.path.exists(video_path):
        os.mkdir(video_path)

    rewards = []
    lengths = []

    out = None
    if not args.no_video:
        env.reset()
        data = env.render()
        out = cv2.VideoWriter(
            os.path.join(video_path, f'policy{idx}_reward{0.0}.mov'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            25,
            (data.shape[1], data.shape[0])
        )

    for i_epi in tqdm(range(args.epi), ncols=80):
        epi_length = 0
        epi_reward = 0
        state = env.reset()
        t = 0
        while True:
            action = cur_policy(state)

            next_state, reward, done, _ = env.step(action)
            epi_length += 1
            epi_reward += reward
            state = next_state
            t += 1

            if not args.no_video:
                if t % 5 == 0:
                    out.write(env.render())
            if done:
                tqdm.write(f'policy: {idx}, epi: {i_epi}, reward: {epi_reward:.2f}, length: {epi_length}')
                rewards.append(epi_reward)
                lengths.append(epi_length)
                break

    print(f'> epi num: {args.epi}, mean reward: {np.mean(rewards):.2f}, std: {np.std(rewards): .2f}')

    if not args.no_video:
        out.release()
        os.rename(os.path.join(video_path, f'policy{idx}_reward{0.0}.mov'),
                  os.path.join(video_path, f'policy{idx}_reward{np.mean(rewards):.2f}_std{np.std(rewards):.2f}.mov'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('-n', '--n-controller', type=int, default=None)
    parser.add_argument('--epi', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-video', action='store_true', default=False)

    # default
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # set up env
    set_seed(args.seed)
    device = torch.device('cpu')
    if args.env is None:
        if args.path is not None:
            settings = read_settings(args.path)
            env = make_env(settings['env'], device=device)
        else:
            raise KeyError('Either the env or the path to the log should be given')
    else:
        env = make_env(args.env, device=device)
        settings = None

    # load models
    if args.path is None:
        env.compute_linearized_controller()
        policy = env.u_nominal
        args.path = f'./logs/{args.env}'
        if not os.path.exists(args.path):
            os.mkdir(args.path)
        policy_id = 0
    else:
        # load demo buffer
        if settings['buffer'] != 'None':
            demo_buffer_size = int(settings['buffer'].split('/')[-1].split('_')[0].split('size')[-1])
            demo_buffer = DemoBuffer(
                buffer_size=demo_buffer_size,
                state_dim=env.n_dims,
                action_dim=env.n_controls,
                device=device,
            )
            demo_buffer.load(settings['buffer'])
        else:
            demo_buffer = None
        model_path = os.path.join(args.path, 'models')

        if settings['algo'] == 'gie-clf' or settings['algo'] == 'clf-sample':
            policy = NeuralCLFController(
                state_dim=env.n_dims,
                action_dim=env.n_controls,
                goal_point=env.goal_point,
                u_eq=env.u_eq,
                state_std=demo_buffer.state_std if demo_buffer is not None else torch.ones(env.n_dims, device=device),
                ctrl_std=demo_buffer.action_std if demo_buffer is not None else torch.ones(env.n_controls,
                                                                                           device=device),
            ).to(device)
            if args.n_controller is not None:
                policy.load(os.path.join(model_path, f'clf_{args.n_controller}.pkl'), device=device)
                policy_id = args.n_controller
            else:
                # load the last controller
                controller_name = os.listdir(model_path)
                controller_name = [i for i in controller_name if 'clf' in i]
                controller_id = sorted([int(i.split('clf_')[1].split('.')[0]) for i in controller_name])
                policy.load(os.path.join(model_path, f'clf_{controller_id[-1]}.pkl'), device=device)
                policy_id = controller_id[-1]
            policy = policy.act
        elif settings['algo'] in AGENTS:
            policy = get_agent(
                algo=settings['algo'],
                state_dim=env.n_dims,
                action_dim=env.n_controls,
                device=device,
                goal_point=env.goal_point,
                u_eq=env.u_eq,
                state_std=demo_buffer.state_std if demo_buffer is not None else torch.ones(env.n_dims, device=device),
                ctrl_std=demo_buffer.action_std if demo_buffer is not None else torch.ones(env.n_controls,
                                                                                           device=device),
            )
            if args.n_controller is not None:
                policy.load(os.path.join(model_path, f'step{args.n_controller}.pkl'), device=device)
                policy_id = args.n_controller
            else:
                # load the last controller
                controller_name = os.listdir(model_path)
                controller_name = [i for i in controller_name if 'step' in i]
                controller_id = sorted([int(i.split('step')[1].split('.')[0]) for i in controller_name])
                policy.load(os.path.join(model_path, f'step{controller_id[-1]}.pkl'), device=device)
                policy_id = controller_id[-1]
            policy = policy.act
        else:
            raise KeyError('Cannot recognize the controller')

    print('> Processing...')
    start_time = time.time()
    execute_policy(policy_id, policy, env, args)
    print(f'> Done in {time.time() - start_time:.0f}s')
