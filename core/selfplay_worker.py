import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


@ray.remote(num_gpus=0.125)
class DataWorker(object):
    def __init__(self, rank, replay_buffer, storage, config):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        """
        self.rank = rank
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer
        # double buffering when data is sufficient
        self.trajectory_pool = []
        self.pool_size = 1
        self.device = self.config.device
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1

    def put(self, data):
        # put a game history into the pool
        self.trajectory_pool.append(data)

    def len_pool(self):
        # current pool size
        return len(self.trajectory_pool)

    def free(self):
        # save the game histories and clear the pool
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool, self.gap_step)
            del self.trajectory_pool[:]

    def put_last_trajectory(self, i, last_game_histories, last_game_priorities, game_histories):
        """put the last game history into the pool if the current game is finished
        Parameters
        ----------
        last_game_histories: list
            list of the last game histories
        last_game_priorities: list
            list of the last game priorities
        game_histories: list
            list of the current game histories
        """
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        last_game_histories[i].game_over()

        self.put((last_game_histories[i], last_game_priorities[i]))
        self.free()

        # reset last block
        last_game_histories[i] = None
        last_game_priorities[i] = None

    def get_priorities(self, i, pred_values_lst, search_values_lst):
        # obtain the priorities at index i
        if self.config.use_priority and not self.config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self):
        # number of parallel mcts
        env_nums = self.config.p_mcts_num
        model = self.config.get_uniform_network()
        model.to(self.device)
        model.eval()

        start_training = False
        envs = [self.config.new_game(self.config.seed + (self.rank + 1) * i) for i in range(env_nums)]

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # 100k benchmark
        total_transitions = 0
        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.num_actors
        with torch.no_grad():
            while True:
                trained_steps = ray.get(self.storage.get_counter.remote())
                # training finished
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    time.sleep(30)
                    break

                init_obses = [env.reset() for env in envs]
                dones = np.array([False for _ in range(env_nums)])
                game_histories = [GameHistory(envs[_].env.action_space, max_length=self.config.history_length,
                                              config=self.config) for _ in range(env_nums)]
                last_game_histories = [None for _ in range(env_nums)]
                last_game_priorities = [None for _ in range(env_nums)]

                # stack observation windows in boundary: s398, s399, s400, current s1 -> for not init trajectory
                stack_obs_windows = [[] for _ in range(env_nums)]

                for i in range(env_nums):
                    stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
                    game_histories[i].init(stack_obs_windows[i])

                # for priorities in self-play
                search_values_lst = [[] for _ in range(env_nums)]
                pred_values_lst = [[] for _ in range(env_nums)]

                # some logs
                eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums)
                step_counter = 0

                self_play_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []
                other_dist = {}

                # play games until max moves
                while not dones.all() and (step_counter <= self.config.max_moves):
                    if not start_training:
                        start_training = ray.get(self.storage.get_start_signal.remote())

                    # get model
                    trained_steps = ray.get(self.storage.get_counter.remote())
                    if trained_steps >= self.config.training_steps + self.config.last_steps:
                        # training is finished
                        time.sleep(30)
                        return
                    if start_training and (total_transitions / max_transitions) > (trained_steps / self.config.training_steps):
                        # self-play is faster than training speed or finished
                        time.sleep(1)
                        continue

                    # set temperature for distributions
                    _temperature = np.array(
                        [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps) for env in
                         envs])

                    # update the models in self-play every checkpoint_interval
                    new_model_index = trained_steps // self.config.checkpoint_interval
                    if new_model_index > self.last_model_index:
                        self.last_model_index = new_model_index
                        # update model
                        weights = ray.get(self.storage.get_weights.remote())
                        model.set_weights(weights)
                        model.to(self.device)
                        model.eval()

                        # log if more than 1 env in parallel because env will reset in this loop.
                        if env_nums > 1:
                            if len(self_play_visit_entropy) > 0:
                                visit_entropies = np.array(self_play_visit_entropy).mean()
                                visit_entropies /= max_visit_entropy
                            else:
                                visit_entropies = 0.

                            if self_play_episodes > 0:
                                log_self_play_moves = self_play_moves / self_play_episodes
                                log_self_play_rewards = self_play_rewards / self_play_episodes
                                log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                            else:
                                log_self_play_moves = 0
                                log_self_play_rewards = 0
                                log_self_play_ori_rewards = 0

                            self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                            log_self_play_ori_rewards, log_self_play_rewards,
                                                                            self_play_rewards_max, _temperature.mean(),
                                                                            visit_entropies, 0,
                                                                            other_dist)
                            self_play_rewards_max = - np.inf

                    step_counter += 1
                    for i in range(env_nums):
                        # reset env if finished
                        if dones[i]:

                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # store current block trajectory
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], priorities))
                            self.free()

                            # reset the finished env and new a env
                            envs[i].close()
                            init_obs = envs[i].reset()
                            game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            last_game_histories[i] = None
                            last_game_priorities[i] = None
                            stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                            game_histories[i].init(stack_obs_windows[i])

                            # log
                            self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_rewards += eps_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1

                            pred_values_lst[i] = []
                            search_values_lst[i] = []
                            # end_tags[i] = False
                            eps_steps_lst[i] = 0
                            eps_reward_lst[i] = 0
                            eps_ori_reward_lst[i] = 0
                            visit_entropies_lst[i] = 0

                    # stack obs for model inference
                    stack_obs = [game_history.step_obs() for game_history in game_histories]
                    if self.config.image_based:
                        stack_obs = prepare_observation_lst(stack_obs)
                        stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                    else:
                        stack_obs = [game_history.step_obs() for game_history in game_histories]
                        stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)

                    if self.config.amp_type == 'torch_amp':
                        with autocast():
                            network_output = model.initial_inference(stack_obs.float())
                    else:
                        network_output = model.initial_inference(stack_obs.float())
                    hidden_state_roots = network_output.hidden_state
                    reward_hidden_roots = network_output.reward_hidden
                    value_prefix_pool = network_output.value_prefix
                    policy_logits_pool = network_output.policy_logits.tolist()

                    roots = cytree.Roots(env_nums, self.config.action_space_size, self.config.num_simulations)
                    noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(env_nums)]
                    roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                    # do MCTS for a policy
                    MCTS(self.config).search(roots, model, hidden_state_roots, reward_hidden_roots)

                    roots_distributions = roots.get_distributions()
                    roots_values = roots.get_values()
                    for i in range(env_nums):
                        deterministic = False
                        if start_training:
                            distributions, value, temperature, env = roots_distributions[i], roots_values[i], _temperature[i], envs[i]
                        else:
                            # before starting training, use random policy
                            value, temperature, env = roots_values[i], _temperature[i], envs[i]
                            distributions = np.ones(self.config.action_space_size)

                        action, visit_entropy = select_action(distributions, temperature=temperature, deterministic=deterministic)
                        obs, ori_reward, done, info = env.step(action)
                        # clip the reward
                        if self.config.clip_reward:
                            clip_reward = np.sign(ori_reward)
                        else:
                            clip_reward = ori_reward

                        # store data
                        game_histories[i].store_search_stats(distributions, value)
                        game_histories[i].append(action, obs, clip_reward)

                        eps_reward_lst[i] += clip_reward
                        eps_ori_reward_lst[i] += ori_reward
                        dones[i] = done
                        visit_entropies_lst[i] += visit_entropy

                        eps_steps_lst[i] += 1
                        total_transitions += 1

                        if self.config.use_priority and not self.config.use_max_priority and start_training:
                            pred_values_lst[i].append(network_output.value[i].item())
                            search_values_lst[i].append(roots_values[i])

                        # fresh stack windows
                        del stack_obs_windows[i][0]
                        stack_obs_windows[i].append(obs)

                        # if game history is full;
                        # we will save a game history if it is the end of the game or the next game history is finished.
                        if game_histories[i].is_full():
                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # calculate priority
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                            # save block trajectory
                            last_game_histories[i] = game_histories[i]
                            last_game_priorities[i] = priorities

                            # new block trajectory
                            game_histories[i] = GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            game_histories[i].init(stack_obs_windows[i])

                for i in range(env_nums):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        # pad over last block trajectory
                        if last_game_histories[i] is not None:
                            self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                        # store current block trajectory
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], priorities))
                        self.free()

                        self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                        self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                        self_play_rewards += eps_reward_lst[i]
                        self_play_ori_rewards += eps_ori_reward_lst[i]
                        self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                        self_play_moves += eps_steps_lst[i]
                        self_play_episodes += 1
                    else:
                        # if the final game history is not finished, we will not save this data.
                        total_transitions -= len(game_histories[i])

                # logs
                visit_entropies = np.array(self_play_visit_entropy).mean()
                visit_entropies /= max_visit_entropy

                if self_play_episodes > 0:
                    log_self_play_moves = self_play_moves / self_play_episodes
                    log_self_play_rewards = self_play_rewards / self_play_episodes
                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                else:
                    log_self_play_moves = 0
                    log_self_play_rewards = 0
                    log_self_play_ori_rewards = 0

                other_dist = {}
                # send logs
                self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                log_self_play_ori_rewards, log_self_play_rewards,
                                                                self_play_rewards_max, _temperature.mean(),
                                                                visit_entropies, 0,
                                                                other_dist)
