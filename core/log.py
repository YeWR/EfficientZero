import ray
import logging

import numpy as np


train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_test')


def _log(config, step_count, log_data, model, replay_buffer, lr, shared_storage, summary_writer, vis_result):
    loss_data, td_data, priority_data = log_data
    total_loss, weighted_loss, loss, reg_loss, policy_loss, value_prefix_loss, value_loss, consistency_loss = loss_data
    if vis_result:
        new_priority, target_value_prefix, target_value, trans_target_value_prefix, trans_target_value, target_value_prefix_phi, target_value_phi, \
        pred_value_prefix, pred_value, target_policies, predicted_policies, state_lst, other_loss, other_log, other_dist = td_data
        batch_weights, batch_indices = priority_data

    replay_episodes_collected, replay_buffer_size, priorities, total_num, worker_logs = ray.get([
        replay_buffer.episodes_collected.remote(), replay_buffer.size.remote(),
        replay_buffer.get_priorities.remote(), replay_buffer.get_total_len.remote(),
        shared_storage.get_worker_logs.remote()])

    worker_ori_reward, worker_reward, worker_reward_max, worker_eps_len, worker_eps_len_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions = worker_logs

    _msg = '#{:<10} Total Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
           'Reward Sum Loss: {:<8.3f} Consistency Loss: {:<8.3f} ] ' \
           'Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Transition Number: {:<8.3f}k ' \
           'Batch Size: {:<10d} Lr: {:<8.3f}'
    _msg = _msg.format(step_count, total_loss, weighted_loss, policy_loss, value_loss, value_prefix_loss, consistency_loss,
                       replay_episodes_collected, replay_buffer_size, total_num / 1000, config.batch_size, lr)
    train_logger.info(_msg)

    if test_dict is not None:
        mean_score = np.mean(test_dict['mean_score'])
        max_score = np.mean(test_dict['max_score'])
        min_score = np.mean(test_dict['min_score'])
        std_score = np.mean(test_dict['std_score'])
        test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                   ''.format(test_counter, config.env_name, mean_score, max_score, min_score, std_score)
        test_logger.info(test_msg)

    if summary_writer is not None:
        if config.debug:
            for name, W in model.named_parameters():
                summary_writer.add_histogram('after_grad_clip' + '/' + name + '_grad', W.grad.data.cpu().numpy(),
                                             step_count)
                summary_writer.add_histogram('network_weights' + '/' + name, W.data.cpu().numpy(), step_count)
            pass
        tag = 'Train'
        if vis_result:
            summary_writer.add_histogram('{}_replay_data/replay_buffer_priorities'.format(tag),
                                         priorities,
                                         step_count)
            summary_writer.add_histogram('{}_replay_data/batch_weight'.format(tag), batch_weights, step_count)
            summary_writer.add_histogram('{}_replay_data/batch_indices'.format(tag), batch_indices, step_count)
            target_value_prefix = target_value_prefix.flatten()
            pred_value_prefix = pred_value_prefix.flatten()
            target_value = target_value.flatten()
            pred_value = pred_value.flatten()
            new_priority = new_priority.flatten()

            summary_writer.add_scalar('{}_statistics/new_priority_mean'.format(tag), new_priority.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/new_priority_std'.format(tag), new_priority.std(), step_count)

            summary_writer.add_scalar('{}_statistics/target_value_prefix_mean'.format(tag), target_value_prefix.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/target_value_prefix_std'.format(tag), target_value_prefix.std(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_value_prefix_mean'.format(tag), pred_value_prefix.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_value_prefix_std'.format(tag), pred_value_prefix.std(), step_count)

            summary_writer.add_scalar('{}_statistics/target_value_mean'.format(tag), target_value.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/target_value_std'.format(tag), target_value.std(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_value_mean'.format(tag), pred_value.mean(), step_count)
            summary_writer.add_scalar('{}_statistics/pre_value_std'.format(tag), pred_value.std(), step_count)

            summary_writer.add_histogram('{}_data_dist/new_priority'.format(tag), new_priority, step_count)
            summary_writer.add_histogram('{}_data_dist/target_value_prefix'.format(tag), target_value_prefix - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/target_value'.format(tag), target_value - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/transformed_target_value_prefix'.format(tag), trans_target_value_prefix,
                                         step_count)
            summary_writer.add_histogram('{}_data_dist/transformed_target_value'.format(tag), trans_target_value,
                                         step_count)
            summary_writer.add_histogram('{}_data_dist/pred_value_prefix'.format(tag), pred_value_prefix - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/pred_value'.format(tag), pred_value - 1e-5, step_count)
            summary_writer.add_histogram('{}_data_dist/pred_policies'.format(tag), predicted_policies.flatten(),
                                         step_count)
            summary_writer.add_histogram('{}_data_dist/target_policies'.format(tag), target_policies.flatten(),
                                         step_count)

            summary_writer.add_histogram('{}_data_dist/hidden_state'.format(tag), state_lst.flatten(), step_count)

            for key, val in other_loss.items():
                if val >= 0:
                    summary_writer.add_scalar('{}_metric/'.format(tag) + key, val, step_count)

            for key, val in other_log.items():
                summary_writer.add_scalar('{}_weight/'.format(tag) + key, val, step_count)

            for key, val in other_dist.items():
                summary_writer.add_histogram('{}_dist/'.format(tag) + key, val, step_count)

        summary_writer.add_scalar('{}/total_loss'.format(tag), total_loss, step_count)
        summary_writer.add_scalar('{}/loss'.format(tag), loss, step_count)
        summary_writer.add_scalar('{}/weighted_loss'.format(tag), weighted_loss, step_count)
        summary_writer.add_scalar('{}/reg_loss'.format(tag), reg_loss, step_count)
        summary_writer.add_scalar('{}/policy_loss'.format(tag), policy_loss, step_count)
        summary_writer.add_scalar('{}/value_loss'.format(tag), value_loss, step_count)
        summary_writer.add_scalar('{}/value_prefix_loss'.format(tag), value_prefix_loss, step_count)
        summary_writer.add_scalar('{}/consistency_loss'.format(tag), consistency_loss, step_count)
        summary_writer.add_scalar('{}/episodes_collected'.format(tag), replay_episodes_collected,
                                  step_count)
        summary_writer.add_scalar('{}/replay_buffer_len'.format(tag), replay_buffer_size, step_count)
        summary_writer.add_scalar('{}/total_node_num'.format(tag), total_num, step_count)
        summary_writer.add_scalar('{}/lr'.format(tag), lr, step_count)

        if worker_reward is not None:
            summary_writer.add_scalar('workers/ori_reward', worker_ori_reward, step_count)
            summary_writer.add_scalar('workers/clip_reward', worker_reward, step_count)
            summary_writer.add_scalar('workers/clip_reward_max', worker_reward_max, step_count)
            summary_writer.add_scalar('workers/eps_len', worker_eps_len, step_count)
            summary_writer.add_scalar('workers/eps_len_max', worker_eps_len_max, step_count)
            summary_writer.add_scalar('workers/temperature', temperature, step_count)
            summary_writer.add_scalar('workers/visit_entropy', visit_entropy, step_count)
            summary_writer.add_scalar('workers/priority_self_play', priority_self_play, step_count)
            for key, val in distributions.items():
                if len(val) == 0:
                    continue

                val = np.array(val).flatten()
                summary_writer.add_histogram('workers/{}'.format(key), val, step_count)

        if test_dict is not None:
            for key, val in test_dict.items():
                summary_writer.add_scalar('train/{}'.format(key), np.mean(val), test_counter)