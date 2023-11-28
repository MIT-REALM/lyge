import argparse
import torch
import cv2
import numpy as np

from tqdm import tqdm

from model.buffer import DemoBuffer
from model.env.env import make_env
from model.utils import set_seed


def render_demo(args):
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    env = make_env(env_id=args.env, device=device)
    set_seed(args.seed)

    # load demonstrations
    if args.buffer_size is None:
        args.buffer_size = int(args.buffer.split('/')[-1].split('_')[0].split('size')[-1])
    demo_buffer = DemoBuffer(
        buffer_size=args.buffer_size,
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        device=device,
    )
    demo_buffer.load(args.buffer)
    print('> Buffer loaded')

    # make gif to render demonstrations
    print('> Rendering demos...')
    env.reset()
    data = env.render()
    out = cv2.VideoWriter(
        args.buffer.split('pkl')[0] + 'mov',
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (data.shape[1], data.shape[0])
    )
    states, actions, _, dones, _ = demo_buffer.all_data
    i_epi = 0
    for i_sample in tqdm(range(env.max_episode_steps * args.epi)):
        if i_sample % 3 == 0:
            im = env.render_demo(
                states[i_sample, :].unsqueeze(0),
                i_sample,
                actions[i_sample, :].unsqueeze(0)
            )
            im.astype(np.uint8)
            out.write(im)
            if dones[i_sample]:
                i_epi += 1
        if i_epi >= args.epi:
            break
    print('> Making videos...')
    out.release()
    print('> Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--buffer', type=str, required=True)

    # default
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--epi', type=int, default=3)

    args = parser.parse_args()
    render_demo(args)
