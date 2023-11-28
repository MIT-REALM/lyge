import os
import torch
import argparse
import time

from model.utils import set_seed, init_logger, eval_policy
from model.env.env import make_env
from model.rl_il.agents import get_agent
from model.trainer.rl_il_trainer import RLILTrainer
from model.trainer.bc_trainer import BCTrainer
from model.buffer import DemoBuffer
from model.controller import NeuralController


def train_rl_il(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    set_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f'> Training with {device}')
    env = make_env(env_id=args.env, device=device)

    # set up logger
    log_path, writer, model_path = init_logger(args.log_path, args.env, args.algo, args.seed, vars(args))

    # load demonstrations
    if args.buffer is None:
        print('> No pre-training')
        demo_buffer = None
        policy_bc = None
    else:
        demo_buffer_size = int(args.buffer.split('/')[-1].split('_')[0].split('size')[-1])
        demo_buffer = DemoBuffer(
            buffer_size=demo_buffer_size,
            state_dim=env.n_dims,
            action_dim=env.n_controls,
            device=device,
        )
        demo_buffer.load(args.buffer)

        # train BC
        print('> Training BC policy...')
        policy_bc = NeuralController(
            state_dim=env.n_dims,
            action_dim=env.n_controls,
            goal_point=env.goal_point,
            u_eq=env.u_eq,
            state_std=demo_buffer.state_std,
            ctrl_std=demo_buffer.action_std,
            hidden_layers=eval(args.hidden_layers_bc),
        ).to(device)
        bc_trainer = BCTrainer(
            policy=policy_bc,
            demo_buffer=demo_buffer,
        )
        bc_trainer.train(n_iter=args.iter_bc, batch_size=args.batchsize_bc)
        policy_bc.save(os.path.join(model_path, f'policy_bc.pkl'))
        policy_bc.disable_grad()
        print('> Done')

        # evaluate BC
        print('> Evaluating BC policy')
        mean_reward = eval_policy(
            env=env,
            policy=policy_bc,
            device=device,
        )
        print('> Done')

    # get agent
    kwargs = {}
    if args.algo == 'drex':
        assert demo_buffer is not None
        assert policy_bc is not None
        kwargs['env'] = env
        kwargs['demo_buffer'] = demo_buffer
        kwargs['policy_bc'] = policy_bc
    elif args.algo == 'airl':
        assert demo_buffer is not None
        kwargs['demo_buffer'] = demo_buffer
    elif args.algo == 'ssrr':
        assert demo_buffer is not None
        kwargs['env'] = env
        kwargs['demo_buffer'] = demo_buffer
        assert args.airl_path is not None
        kwargs['airl_path'] = args.airl_path
    agent = get_agent(
        algo=args.algo,
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        device=device,
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=demo_buffer.state_std if demo_buffer is not None else torch.ones(env.n_dims, device=device),
        ctrl_std=demo_buffer.action_std if demo_buffer is not None else torch.ones(env.n_controls, device=device),
        kwargs=kwargs
    )
    if policy_bc is not None:
        agent.set_controller(policy_bc.net)
    agent.save(os.path.join(model_path, f'step-1.pkl'))

    # setup RL trainer
    trainer = RLILTrainer(
        env=env,
        agent=agent,
        writer=writer,
        model_dir=model_path,
        num_steps=args.steps,
        eval_interval=args.steps // 50,
    )
    print(f'> Training {args.algo.upper()}...')
    start_time = time.time()
    trainer.train()
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--buffer', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--airl-path', type=str, default=None)

    # default
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter-bc', type=int, default=5000)
    parser.add_argument('--batchsize-bc', type=int, default=256)
    parser.add_argument('--hidden-layers-bc', type=str, default='[128, 128]')
    parser.add_argument('--log-path', type=str, default='./logs')
    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()
    train_rl_il(args)
