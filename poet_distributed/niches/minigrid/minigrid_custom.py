from collections import namedtuple
from gym-minigrid.envs.mazeEnv import MazeEnv
from gym.utils import seeding

Env_config = namedtuple('Env_config', [
    'name',
    'lava_prob',
    'obstacle_lvl',
    'box_to_ball_prob',
    'door_prob',
    'wall_prob',
])


# Create a modified version of gym-minigrid that uses recursive division to create mazes
#  and mutable parameters to modify object placements, amount of walls, etc
class MinigridCustom(MazeEnv):
    def __repr__(self):
        return "{}\n{}".format(self.__dict__, self.__dict__["np_random"].get_state())

    def __init__(self, env_config):
        self.set_env_config(env_config)
        self._seed()
        
        super().__init__(self, size=6, limit=1, 
        

    def _seed(self,seed=none)
        self.env_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_env_config(env_config):
        self.config = env_config


    def reset(self):
        # Get specific probabilities from the env_config ranges
        self.lava_prob = self.np_random.randfloat(*self.config.lava_prob)
        self.obstacle_level = self.np_random.randfloat(*self.config.obstacle_level)
        self.box2ball = self.np_random.randfloat(*self.config.box_to_ball_prob)
        self.door_prob = self.np_random.randfloat(*self.config.door_prob)
        self.wall_prob = self.np_random.randfloat(*self.config.wall_prob)

	super().reset(self)
