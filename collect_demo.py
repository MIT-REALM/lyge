import numpy as np
import argparse
import torch
import os
import cv2

from tqdm import tqdm

from model.buffer import DemoBuffer
from model.utils import set_seed, read_settings
from model.env.env import make_env
from model.rl_il.agents import AGENTS, get_agent


def collect_demo(args):
    set_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    env = make_env(env_id=args.env, device=device)
    if env.use_lqr:
        env.compute_linearized_controller()

    buffer = DemoBuffer(
        buffer_size=args.steps,
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        device=device
    )

    if not os.path.exists(args.path):
        os.mkdir(args.path)
    if not os.path.exists(os.path.join(args.path, args.env)):
        os.mkdir(os.path.join(args.path, args.env))

    # load RL agent
    if args.agent_path is not None:
        settings = read_settings(args.agent_path)
        assert settings['algo'] in AGENTS
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
        model_path = os.path.join(args.agent_path, 'models')
        if args.n_controller is not None:
            policy.load(os.path.join(model_path, f'step{args.n_controller}.pkl'), device=device)
        else:
            # load the last controller
            controller_name = os.listdir(model_path)
            controller_name = [i for i in controller_name if 'step' in i]
            controller_id = sorted([int(i.split('step')[1].split('.')[0]) for i in controller_name])
            policy.load(os.path.join(model_path, f'step{controller_id[-1]}.pkl'), device=device)
        controller = policy.act
    else:
        controller = env.u_nominal

    epi_rewards = []
    cur_reward = 0
    cur_step = 0

    out = None
    video_path = os.path.join(args.path, args.env)
    if args.render:
        env.reset()
        data = env.render()
        out = cv2.VideoWriter(
            os.path.join(video_path, f'0.mov'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            60,
            (data.shape[1], data.shape[0])
        )

    state = env.reset()
    for i in tqdm(range(args.steps)):
        action = controller(state)
        action += torch.randn_like(action) * args.std * (env.control_limits[0] - env.control_limits[1])
        action += args.bias
        next_state, reward, done, _ = env.step(action)
        cur_reward += reward
        buffer.append(state, action, reward, done, next_state)
        state = next_state
        cur_step += 1

        if args.render:
            out.write(env.render())

        if done:
            tqdm.write(f'epi: {len(epi_rewards)}, reward: {cur_reward:.0f}, steps: {cur_step}')
            state = env.reset()
            epi_rewards.append(cur_reward)
            cur_reward = 0
            cur_step = 0

    buffer_name = f'size{args.steps}_std{args.std}_bias{args.bias}_reward{np.mean(epi_rewards):.0f}.pkl'
    buffer.save(os.path.join(args.path, args.env, buffer_name))
    print(f'> buffer saved, mean reward: {np.mean(epi_rewards):.2f}, std: {np.std(epi_rewards):.2f}')

    if args.render:
        out.release()
        video_name = f'size{args.steps}_std{args.std}_bias{args.bias}_reward{np.mean(epi_rewards):.0f}.mov'
        os.rename(os.path.join(video_path, f'0.mov'), os.path.join(video_path, video_name))
    print('> video saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--agent-path', type=str, default=None)
    parser.add_argument('--n-controller', type=int, default=None)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--bias', type=float, default=0.1)
    parser.add_argument('--render', action='store_true', default=False)

    # default
    parser.add_argument('--path', type=str, default='./demos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--steps', type=int, default=20000)

    args = parser.parse_args()
    collect_demo(args)
