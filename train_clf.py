import argparse
import copy
import torch
import os
import time
import yaml

from model.utils import set_seed, init_logger, eval_policy, collect_demo
from model.buffer import DemoBuffer
from model.trainer.bc_trainer import BCTrainer
from model.trainer.clf_trainer import CLFTrainer
from model.trainer.env_model_trainer import EnvModelTrainer
from model.controller import NeuralController, NeuralCLFController
from model.env.env import make_env
from model.env_model.safe_affine import NeuralSafeCriticalAffineEnv
from model.env_model.safe_critical import NeuralSafeCriticalEnv
from model.rl_il.agents import AIRL


def train_clf(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    set_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f'> Training with {device}')
    env = make_env(env_id=args.env, device=device)

    # load hyper-parameters
    cur_path = os.getcwd()
    if os.path.exists(os.path.join(cur_path, 'model/env/hyperparams', f'{args.env}.yaml')):
        print('> Using tuned hyper-parameters')
        with open(os.path.join(cur_path, 'model/env/hyperparams', f'{args.env}.yaml')) as f:
            hyper_params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise KeyError(f'Cannot find hyper-parameters for {args.env}. '
                       f'Please put {args.env}.yaml in model/env/hyperparams to specify hyper-parameters!')

    # ablation hyper-parameters
    if args.eps is not None:
        hyper_params['loss_eps']['decrease'] = args.eps
    if args.ctrl_weight is not None:
        hyper_params['loss_coefs']['control'] = args.ctrl_weight

    # debug mode
    if args.debug:
        args.iter_model = 500
        args.iter_clf = 100
        args.iter_bc = 500

    # set up logger
    if args.sample_all:
        if args.n_sample is None:
            algo_name = 'clf-sample'
        else:
            algo_name = 'clf-dense'
    else:
        algo_name = 'gie-clf'
    log_path, writer, model_path = init_logger(args.log_path, args.env, algo_name, args.seed, vars(args), hyper_params)

    # load demonstrations
    demo_buffer_size = int(args.buffer.split('/')[-1].split('_')[0].split('size')[-1])
    demo_buffer = DemoBuffer(
        buffer_size=demo_buffer_size,
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        device=device,
    )
    demo_buffer.load(args.buffer)
    if args.buffer_size is None:
        args.buffer_size = demo_buffer_size

    if args.airl is None:
        # train BC
        print('> Training BC policy...')
        policy_bc = NeuralController(
            state_dim=env.n_dims,
            action_dim=env.n_controls,
            goal_point=env.goal_point,
            u_eq=env.u_eq,
            state_std=torch.ones(env.n_dims, device=device),  # demo_buffer.state_std,
            ctrl_std=torch.ones(env.n_controls, device=device),  # demo_buffer.action_std,
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
    else:
        # load pre-trained AIRL policy
        agent = AIRL(
            state_dim=env.n_dims,
            action_dim=env.n_controls,
            device=device,
            goal_point=env.goal_point,
            u_eq=env.u_eq,
            state_std=demo_buffer.state_std,
            ctrl_std=demo_buffer.action_std,
            demo_buffer=demo_buffer
        )
        agent.load(args.airl, device=device)
        print('> AIRL loaded')

        # evaluate AIRL
        print('> Evaluating AIRL policy')
        mean_reward = eval_policy(env, agent, device)
        print('> Done')

    env_model = NeuralSafeCriticalEnv(
        base_env=env,
        device=device,
        safe_buffer=demo_buffer,
        goal_relaxation=0.01,
        normalize=True,
        sparse_g=False
    )

    clf_controller = NeuralCLFController(
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=demo_buffer.state_std,
        ctrl_std=demo_buffer.action_std,
        clf_lambda=args.clf_lambda
    ).to(device)
    if args.airl is not None:
        clf_controller.controller.net.set_weight(agent.actor.net.net)
    else:
        clf_controller.set_controller(policy_bc)
    clf_controller.save(os.path.join(model_path, f'clf_{-1}.pkl'))

    # reference controller
    ref_controller = NeuralCLFController(
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=demo_buffer.state_std,
        ctrl_std=demo_buffer.action_std,
        clf_lambda=args.clf_lambda
    ).to(device)
    ref_controller.disable_grad()

    # collect training data if we use samples from the whole state space
    if args.sample_all:
        print('> Collecting training data...')
        if args.n_sample is None:
            data_size = demo_buffer.buffer_size + args.buffer_size * args.iter_outer
        else:
            data_size = args.n_sample
        demo_buffer = DemoBuffer(
            buffer_size=data_size,
            state_dim=env.n_dims,
            action_dim=env.n_controls,
            device=device,
        )
        demo_buffer = collect_demo(
            buffer=demo_buffer,
            env=env,
            policy='random',
            n_steps=data_size,
            sample_traj=False
        )
        print('> Done')
        demo_buffer.save(os.path.join(model_path, f'data_size{demo_buffer.buffer_size}.pkl'))
        print('> Training data saved')

    start_time = time.time()
    for i_outer in range(1, args.iter_outer):
        iter_start_time = time.time()
        print(f'> ----------- Outer iter: {i_outer} -----------')
        # fit local model
        print('> Training environment model...')
        env_model_trainer = EnvModelTrainer(
            env_model=env_model,
            env=env,
            demo_buffer=demo_buffer,
            writer=writer
        )
        env_model_trainer.train(
            n_iter=args.iter_model,
            batch_size=args.batchsize_model,
            sample_demo=True,
            outer_iter=i_outer
        )
        env_model.save(os.path.join(model_path, f'env_model_{i_outer}.pkl'))
        print('> Done')

        print('> Training CLF controller...')
        # ref_controller = copy.deepcopy(clf_controller)
        # ref_controller.disable_grad()
        ref_controller.load_state_dict(clf_controller.state_dict())
        if i_outer < args.ctrl_training_start:
            clf_controller.disable_grad_ctrl()
        else:
            clf_controller.enable_grad()
        if args.gt_baseline:
            baseline = env.u_nominal
        else:
            if args.airl is not None:
                baseline = agent.act  # policy_bc
            else:
                baseline = policy_bc

        # decrease controller loss coefficient
        if 'control_decrease' in hyper_params['loss_coefs'].keys():
            control_decrease_interval = args.iter_outer // hyper_params['loss_coefs']['control_decrease']
            if i_outer % control_decrease_interval == 0:
                hyper_params['loss_coefs']['control'] /= 2

        clf_trainer = CLFTrainer(
            controller=clf_controller,
            controller_ref=ref_controller,
            baseline=baseline if i_outer < args.use_baseline else None,
            env=env_model,
            demo_buffer=demo_buffer,
            writer=writer,
            hyper_params=hyper_params,
            # gt_env=env,
        )
        clf_trainer.train(
            n_iter=args.iter_clf,
            batch_size=args.batchsize_clf,
            sample_demo=True,
            outer_iter=i_outer
        )
        clf_controller.save(os.path.join(model_path, f'clf_{i_outer}.pkl'))

        # evaluate CLF controller
        print('> Evaluating policy...')
        mean_reward = eval_policy(
            env=env,
            policy=clf_controller,
            device=device,
        )
        writer.add_scalar('eval/reward', mean_reward.item(), i_outer)
        print('> Done')

        # collect demos
        if not args.sample_all:
            print('> Collecting demos...')
            demo_buffer.expand(demo_buffer.buffer_size + args.buffer_size)
            # trained_controller = copy.deepcopy(clf_controller)
            # trained_controller.disable_grad()
            demo_buffer = collect_demo(
                buffer=demo_buffer,
                env=env,
                policy=clf_controller,
                n_steps=args.buffer_size,
                sample_traj=True
            )
            writer.add_scalar('eval/demo reward', demo_buffer.mean_reward, i_outer)
            print('> Done')
            if os.path.exists(
                    os.path.join(model_path, f'final_demo_size{demo_buffer.buffer_size - args.buffer_size}.pkl')):
                os.remove(os.path.join(model_path, f'final_demo_size{demo_buffer.buffer_size - args.buffer_size}.pkl'))
            demo_buffer.save(os.path.join(model_path, f'final_demo_size{demo_buffer.buffer_size}.pkl'))
            print('> Final demo saved')
        iter_end_time = time.time()
        print(f'> Iter time: {iter_end_time - iter_start_time:.0f}s, total time: {iter_end_time - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--buffer', type=str, required=True)
    parser.add_argument('--airl', type=str, default=None)
    parser.add_argument('--sample-all', action='store_true', default=False,
                        help='Sample from the whole state space while training')
    parser.add_argument('--n-sample', type=int, default=None,
                        help='number of samples to collect when sample_all is activated')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--ctrl-training-start', type=int, default=0)
    parser.add_argument('--iter-outer', type=int, default=50)
    parser.add_argument('--iter-model', type=int, default=5000)
    parser.add_argument('--iter-clf', type=int, default=2000)
    parser.add_argument('--use-baseline', type=int, default=30)
    parser.add_argument('--gt-baseline', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    # ablation
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--clf-lambda', type=float, default=1.0)
    parser.add_argument('--ctrl-weight', type=float, default=None)

    # default
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter-bc', type=int, default=5000)
    parser.add_argument('--batchsize-bc', type=int, default=256)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--buffer-size', type=int, default=8000)
    parser.add_argument('--batchsize-model', type=int, default=1024)
    parser.add_argument('--batchsize-clf', type=int, default=512)
    parser.add_argument('--log-path', type=str, default='./logs')
    parser.add_argument('--hidden-layers-lyapunov', type=str, default='[128, 128]')
    parser.add_argument('--hidden-layers-bc', type=str, default='[128, 128]')
    parser.add_argument('--lr-lyapunov', type=float, default=3e-4)

    args = parser.parse_args()
    train_clf(args)
