import os
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.env.base import ControlAffineSystem
from model.rl_il.base import Agent


class RLILTrainer:

    def __init__(
            self,
            env: ControlAffineSystem,
            agent: Agent,
            writer: SummaryWriter,
            model_dir: str,
            num_steps: int = 10**5,
            eval_interval: int = 10**4,
            num_eval_episodes: int = 5
    ):
        self.env = env
        self.agent = agent
        self.writer = writer
        self.model_dir = model_dir

        # parameters
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # episode's time step
        t = 0

        state = self.env.reset()

        for step in tqdm(range(1, self.num_steps + 1), ncols=80):
            # pass to the algorithm to update state and episode time step
            state, t = self.agent.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.agent.is_update(step):
                self.agent.update()

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.agent.save(os.path.join(self.model_dir, f'step{step}.pkl'))

    def evaluate(self, step: int):
        returns = []

        for _ in range(self.num_eval_episodes):
            state = self.env.reset()
            episode_return = 0.0
            t = 0

            while t < self.env.max_episode_steps:
                action = self.agent.act(state)
                state, reward, done, _ = self.env.step(action)
                episode_return += reward
                t += 1
                if done:
                    break

            returns.append(episode_return)

        self.writer.add_scalar('eval/reward', np.mean(returns), step)
        tqdm.write(f'Num steps: {step:<6}, '
                   f'Return: {np.mean(returns):<5.1f}, '
                   f'Min/Max Return: {np.min(returns):<5.1f}/{np.max(returns):<5.1f}')
