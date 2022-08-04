import os
import ray
import time
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from core.log import _log
from core.test import _test
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage, QueueStorage
from core.selfplay_worker import DataWorker
from core.reanalyze_worker import BatchWorker_GPU, BatchWorker_CPU


def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count):
    # adjust learning rate, step lr every lr_decay_steps
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result=False):
    """update models given a batch data
    Parameters
    ----------
    model: Any
        EfficientZero models
    batch: Any
        a batch data inlcudes [inputs_batch, targets_batch]
    replay_buffer: Any
        replay buffer
    scaler: Any
        scaler for torch amp
    vis_result: bool
        True -> log some visualization data in tensorboard (some distributions, values, etc)
    """
    inputs_batch, targets_batch = batch
    obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
    target_value_prefix, target_value, target_policy = targets_batch

    # [:, 0: config.stacked_observations * 3,:,:]
    # obs_batch_ori is the original observations in a batch
    # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
    # obs_target_batch is the observations for s_t (hidden states from representation function)
    # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float() / 255.0
    obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
    obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]

    # do augmentations
    if config.use_augmentation:
        obs_batch = config.transform(obs_batch)
        obs_target_batch = config.transform(obs_target_batch)

    # use GPU tensor
    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_value_prefix = torch.from_numpy(target_value_prefix).to(config.device).float()
    target_value = torch.from_numpy(target_value).to(config.device).float()
    target_policy = torch.from_numpy(target_policy).to(config.device).float()
    weights = torch.from_numpy(weights_lst).to(config.device).float()

    batch_size = obs_batch.size(0)
    assert batch_size == config.batch_size == target_value_prefix.size(0)
    metric_loss = torch.nn.L1Loss()

    # some logs preparation
    other_log = {}
    other_dist = {}

    other_loss = {
        'l1': -1,
        'l1_1': -1,
        'l1_-1': -1,
        'l1_0': -1,
    }
    for i in range(config.num_unroll_steps):
        key = 'unroll_' + str(i + 1) + '_l1'
        other_loss[key] = -1
        other_loss[key + '_1'] = -1
        other_loss[key + '_-1'] = -1
        other_loss[key + '_0'] = -1

    # transform targets to categorical representation
    transformed_target_value_prefix = config.scalar_transform(target_value_prefix)
    target_value_prefix_phi = config.reward_phi(transformed_target_value_prefix)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)

    if config.amp_type == 'torch_amp':
        with autocast():
            value, _, policy_logits, hidden_state, reward_hidden = model.initial_inference(obs_batch)
    else:
        value, _, policy_logits, hidden_state, reward_hidden = model.initial_inference(obs_batch)
    scaled_value = config.inverse_value_transform(value)

    if vis_result:
        state_lst = hidden_state.detach().cpu().numpy()

    predicted_value_prefixs = []
    # Note: Following line is just for logging.
    if vis_result:
        predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(policy_logits, dim=1).detach().cpu()

    # calculate the new priorities for each transition
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps

    # loss of the first step
    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    value_prefix_loss = torch.zeros(batch_size, device=config.device)
    consistency_loss = torch.zeros(batch_size, device=config.device)

    target_value_prefix_cpu = target_value_prefix.detach().cpu()
    gradient_scale = 1 / config.num_unroll_steps
    # loss of the unrolled steps
    if config.amp_type == 'torch_amp':
        # use torch amp
        with autocast():
            for step_i in range(config.num_unroll_steps):
                # unroll with the dynamics function
                value, value_prefix, policy_logits, hidden_state, reward_hidden = model.recurrent_inference(hidden_state, reward_hidden, action_batch[:, step_i])

                beg_index = config.image_channel * step_i
                end_index = config.image_channel * (step_i + config.stacked_observations)

                # consistency loss
                if config.consistency_coeff > 0:
                    # obtain the oracle hidden states from representation function
                    _, _, _, presentation_state, _ = model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                    # no grad for the presentation_state branch
                    dynamic_proj = model.project(hidden_state, with_grad=True)
                    observation_proj = model.project(presentation_state, with_grad=False)
                    temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                    other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                    consistency_loss += temp_loss

                policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
                value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
                value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
                # Follow MuZero, set half gradient
                hidden_state.register_hook(lambda grad: grad * 0.5)

                # reset hidden states
                if (step_i + 1) % config.lstm_horizon_len == 0:
                    reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device),
                                     torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device))

                if vis_result:
                    scaled_value_prefixs = config.inverse_reward_transform(value_prefix.detach())
                    scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                    predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                    predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                    predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                    state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                    key = 'unroll_' + str(step_i + 1) + '_l1'

                    value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                    value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                    value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                    target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                    other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                    if value_prefix_indices_1.any():
                        other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
                    if value_prefix_indices_n1.any():
                        other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
                    if value_prefix_indices_0.any():
                        other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])
    else:
        for step_i in range(config.num_unroll_steps):
            # unroll with the dynamics function
            value, value_prefix, policy_logits, hidden_state, reward_hidden = model.recurrent_inference(hidden_state, reward_hidden, action_batch[:, step_i])

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)

            # consistency loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                _, _, _, presentation_state, _ = model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                # no grad for the presentation_state branch
                dynamic_proj = model.project(hidden_state, with_grad=True)
                observation_proj = model.project(presentation_state, with_grad=False)
                temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
            value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
            # Follow MuZero, set half gradient
            hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % config.lstm_horizon_len == 0:
                reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device),
                                 torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device))

            if vis_result:
                scaled_value_prefixs = config.inverse_reward_transform(value_prefix.detach())
                scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
                if value_prefix_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
                if value_prefix_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])
    # ----------------------------------------------------------------------------------
    # weighted loss with masks (some invalid states which are out of trajectory.)
    loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss +
            config.value_loss_coeff * value_loss + config.reward_loss_coeff * value_prefix_loss)
    weighted_loss = (weights * loss).mean()

    # backward
    parameters = model.parameters()
    if config.amp_type == 'torch_amp':
        with autocast():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
    else:
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
    optimizer.zero_grad()

    if config.amp_type == 'none':
        total_loss.backward()
    elif config.amp_type == 'torch_amp':
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if config.amp_type == 'torch_amp':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # ----------------------------------------------------------------------------------
    # update priority
    new_priority = value_priority
    replay_buffer.update_priorities.remote(indices, new_priority, make_time)

    # packing data for logging
    loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                 value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean())
    if vis_result:
        reward_w_dist, representation_mean, dynamic_mean, reward_mean = model.get_params_mean()
        other_dist['reward_weights_dist'] = reward_w_dist
        other_log['representation_weight'] = representation_mean
        other_log['dynamic_weight'] = dynamic_mean
        other_log['reward_weight'] = reward_mean

        # reward l1 loss
        value_prefix_indices_0 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
        value_prefix_indices_n1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
        value_prefix_indices_1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

        target_value_prefix_base = target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

        predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
        predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
        other_loss['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
        if value_prefix_indices_1.any():
            other_loss['l1_1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
        if value_prefix_indices_n1.any():
            other_loss['l1_-1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
        if value_prefix_indices_0.any():
            other_loss['l1_0'] = metric_loss(predicted_value_prefixs[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])

        td_data = (new_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                   transformed_target_value_prefix.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                   target_value_prefix_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                   predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                   target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst,
                   other_loss, other_log, other_dist)
        priority_data = (weights, indices)
    else:
        td_data, priority_data = None, None

    return loss_data, td_data, priority_data, scaler


def _train(model, target_model, replay_buffer, shared_storage, batch_storage, config, summary_writer):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    replay_buffer: Any
        replay buffer
    shared_storage: Any
        model storage
    batch_storage: Any
        batch storage (queue)
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    model = model.to(config.device)
    target_model = target_model.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)

    scaler = GradScaler()

    model.train()
    target_model.eval()
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation:
        config.set_transforms()

    # wait until collecting enough data to start
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_transitions):
        time.sleep(1)
        pass
    print('Begin training...')
    # set signals for other workers
    shared_storage.set_start_signal.remote()

    step_count = 0
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = model.get_weights()

    # while loop
    while step_count < config.training_steps + config.last_steps:
        # remove data if the replay buffer is full. (more data settings)
        if step_count % 1000 == 0:
            replay_buffer.remove_to_fit.remote()

        # obtain a batch
        batch = batch_storage.pop()
        if batch is None:
            time.sleep(0.3)
            continue
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count)

        # update model for self-play
        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())

        # update model for reanalyzing
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights)
            recent_weights = model.get_weights()

        if step_count % config.vis_interval == 0:
            vis_result = True
        else:
            vis_result = False

        if config.amp_type == 'torch_amp':
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result)
            scaler = log_data[3]
        else:
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result)

        if step_count % config.log_interval == 0:
            _log(config, step_count, log_data[0:3], model, replay_buffer, lr, shared_storage, summary_writer, vis_result)

        # The queue is empty.
        if step_count >= 100 and step_count % 50 == 0 and batch_storage.get_len() == 0:
            print('Warning: Batch Queue is empty (Require more batch actors Or batch actor fails).')

        step_count += 1

        # save models
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(model.get_weights())
    time.sleep(30)
    return model.get_weights()


def train(config, summary_writer, model_path=None):
    """training process
    Parameters
    ----------
    summary_writer: Any
        logging for tensorboard
    model_path: str
        model path for resuming
        default: train from scratch
    """
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    storage = SharedStorage.remote(model, target_model)

    # prepare the batch and mctc context storage
    batch_storage = QueueStorage(15, 20)
    mcts_storage = QueueStorage(18, 25)
    replay_buffer = ReplayBuffer.remote(config=config)

    # other workers
    workers = []

    # reanalyze workers
    cpu_workers = [BatchWorker_CPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.cpu_actor)]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    gpu_workers = [BatchWorker_GPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.gpu_actor)]
    workers += [gpu_worker.run.remote() for gpu_worker in gpu_workers]

    # self-play workers
    data_workers = [DataWorker.remote(rank, replay_buffer, storage, config) for rank in range(0, config.num_actors)]
    workers += [worker.run.remote() for worker in data_workers]
    # test workers
    workers += [_test.remote(config, storage)]

    # training loop
    final_weights = _train(model, target_model, replay_buffer, storage, batch_storage, config, summary_writer)

    ray.wait(workers)
    print('Training over...')

    return model, final_weights
