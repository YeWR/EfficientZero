import os
import cv2
import gym
import torch
import random
import shutil
import logging

import numpy as np

from scipy.stats import entropy
from baselines.common.atari_wrappers import NoopResetEnv, TimeLimit


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


def make_atari(env_id, skip=4, max_episode_steps=None):
    """Make Atari games from baselines
    Parameters
    ----------
    env_id: str
        name of environment
    skip: int
        frame skip
    max_episode_steps: int
        max moves for an episode
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    # initialize the logger
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def select_action(visit_counts, temperature=1, deterministic=True):
    """select action from the root visit counts.
    Parameters
    ----------
    temperature: float
        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy


def prepare_observation_lst(observation_lst):
    """Prepare the observations to satisfy the input fomat of torch
    [B, S, W, H, C] -> [B, S x C, W, H]
    batch, stack num, width, height, channel
    """
    # B, S, W, H, C
    observation_lst = np.array(observation_lst, dtype=np.uint8)
    observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def arr_to_str(arr):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str


def str_to_arr(s, gray_scale=False):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function decodes the observation numpy arr from the jpeg strings
    Parameters
    ----------
    s: string
        the inputs
    gray_scale: bool
        True -> the inputs observation is gray not RGB.
    """
    nparr = np.frombuffer(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr
