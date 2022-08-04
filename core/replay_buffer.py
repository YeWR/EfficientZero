import ray
import time

import numpy as np


@ray.remote
class ReplayBuffer(object):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """
    def __init__(self, config=None):
        self.config = config
        self.batch_size = config.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = config.priority_prob_alpha
        self.transition_top = int(config.transition_num * 10 ** 6)
        self.clear_time = 0

    def save_pools(self, pools, gap_step):
        # save a list of game histories
        for (game, priorities) in pools:
            # Only append end game
            # if end_tag:
            if len(game) > 0:
                self.save_game(game, True, gap_step, priorities)

    def save_game(self, game, end_tag, gap_steps, priorities=None):
        """Save a game history block
        Parameters
        ----------
        game: Any
            a game history block
        end_tag: bool
            True -> the game is finished. (always True)
        gap_steps: int
            if the game is not finished, we only save the transitions that can be computed
        priorities: list
            the priorities corresponding to the transitions in the game history
        """
        if self.get_total_len() >= self.config.total_transitions:
            return

        if end_tag:
            self._eps_collected += 1
            valid_len = len(game)
        else:
            valid_len = len(game) - gap_steps

        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(game))]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            priorities = priorities.copy().reshape(-1)
            # priorities[valid_len:len(game)] = 0.
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]

    def get_game(self, idx):
        # return a game
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_pos_lst:           transition index in game (relative index)
        indices_lst:            transition index in replay buffer
        weights_lst:            the weight concering the priority
        make_time:              the time the batch is made (for correctly updating replay buffer when data is deleted)
        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority
        """
        assert beta > 0

        total = self.get_total_len()

        probs = self.priorities ** self._alpha

        probs /= probs.sum()
        # sample data
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
        return context

    def update_priorities(self, batch_indices, batch_priorities, make_time):
        # update the priorities for data still in replay buffer
        for i in range(len(batch_indices)):
            if make_time[i] > self.clear_time:
                idx, prio = batch_indices[i], batch_priorities[i]
                self.priorities[idx] = prio

    def remove_to_fit(self):
        # remove some old data if the replay buffer is full.
        current_size = self.size()
        total_transition = self.get_total_len()
        if total_transition > self.transition_top:
            index = 0
            for i in range(current_size):
                total_transition -= len(self.buffer[i])
                if total_transition <= self.transition_top * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self.config.batch_size:
                self._remove(index + 1)

    def _remove(self, num_excess_games):
        # delete game histories
        excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        del self.game_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def clear_buffer(self):
        del self.buffer[:]

    def size(self):
        # number of games
        return len(self.buffer)

    def episodes_collected(self):
        # number of collected histories
        return self._eps_collected

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_total_len(self):
        # number of transitions
        return len(self.priorities)
