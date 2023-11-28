import gym


gym.envs.registration.register(
     id='InvertedTriplePendulum-v0',
     entry_point='model.env.cart_triple_pole:InvertedTriplePendulumEnv',
     max_episode_steps=1000,
)
