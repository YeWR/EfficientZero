import argparse
import logging.config
import os

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir, set_seed
if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['atari'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--amp_type', required=True, choices=['torch_amp', 'none'],
                        help='choose automated mixed precision type')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--save_video', action='store_true', default=False, help='save video in test.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int, default=14, help='batch cpu actor')
    parser.add_argument('--gpu_actor', type=int, default=20, help='batch bpu actor')
    parser.add_argument('--p_mcts_num', type=int, default=4, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int, default=4, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=80, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_root_value', action='store_true', default=False,
                        help='choose to use root value in reanalyzing')
    parser.add_argument('--use_priority', action='store_true', default=False,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=False, help='max priority')
    parser.add_argument('--test_episodes', type=int, default=10, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none', help='debug string')
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
    parser.add_argument('--object_store_memory', type=int, default=150 * 1024 * 1024 * 1024, help='object store memory')

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'

    if args.opr == 'train':
        ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus,
                 object_store_memory=args.object_store_memory)
    else:
        ray.init()

    # seeding random iterators
    set_seed(args.seed)

    # import corresponding configuration , neural networks and envs
    if args.case == 'atari':
        from config.atari import game_config
    else:
        raise Exception('Invalid --case option')

    # set config as per arguments
    exp_path = game_config.set_config(args)
    exp_path, log_base_path = make_results_dir(exp_path, args)

    # set-up logger
    init_logger(log_base_path)
    logging.getLogger('train').info('Path: {}'.format(exp_path))
    logging.getLogger('train').info('Param: {}'.format(game_config.get_hparams()))

    device = game_config.device
    try:
        if args.opr == 'train':
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            if args.load_model and os.path.exists(args.model_path):
                model_path = args.model_path
            else:
                model_path = None
            model, weights = train(game_config, summary_writer, model_path)
            model.set_weights(weights)
            total_steps = game_config.training_steps + game_config.last_steps
            test_score, _, test_path = test(game_config, model.to(device), total_steps, game_config.test_episodes, device, render=False, save_video=args.save_video, final_test=True, use_pb=True)
            mean_score = test_score.mean()
            std_score = test_score.std()

            test_log = {
                'mean_score': mean_score,
                'std_score': std_score,
            }
            for key, val in test_log.items():
                summary_writer.add_scalar('train/{}'.format(key), np.mean(val), total_steps)

            test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                       ''.format(total_steps, game_config.env_name, mean_score, test_score.max(), test_score.min(), std_score)
            logging.getLogger('train_test').info(test_msg)
            if args.save_video:
                logging.getLogger('train_test').info('Saving video in path: {}'.format(test_path))
        elif args.opr == 'test':
            assert args.load_model
            if args.model_path is None:
                model_path = game_config.model_path
            else:
                model_path = args.model_path
            assert os.path.exists(model_path), 'model not found at {}'.format(model_path)

            model = game_config.get_uniform_network().to(device)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            test_score, _, test_path = test(game_config, model, 0, args.test_episodes, device=device, render=args.render, save_video=args.save_video, final_test=True, use_pb=True)
            mean_score = test_score.mean()
            std_score = test_score.std()
            logging.getLogger('test').info('Test Mean Score: {} (max: {}, min: {})'.format(mean_score, test_score.max(), test_score.min()))
            logging.getLogger('test').info('Test Std Score: {}'.format(std_score))
            if args.save_video:
                logging.getLogger('test').info('Saving video in path: {}'.format(test_path))
        else:
            raise Exception('Please select a valid operation(--opr) to be performed')
        ray.shutdown()
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
