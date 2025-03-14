from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import csv
import shutil
import torch
import numpy as np
import itertools
import pdb
from termcolor import colored


ATARI_TRAJ_METRICS = {
                'true_episode_reward': 'TR',
                'episode_reward_hat': 'RHat',
                'true_raw_episode_reward': 'TRR',
                'episode_length': 'LEN',
               }

HIGHWAY_TRAJ_METRICS = {
                'total_reward': 'REWARD',
                'cnt_step': 'TOTAL_STEP',
                'average_speed': 'SPEED_AVG',
                'average_forward_speed': 'F_SPEED_AVG',
                'crashed': 'CRASH',
                'truncated': 'TRUNC',
                'total_time': 'TOTAL_TIME',
                'total_distance': 'TOTAL_DIST',
                'cnt_lane_changed': 'LANE_CHANGE',
                'total_off_road_time': 'OFF_ROAD_TIME',
                'total_collision_reward': 'COLLISION_REW',
                'total_non_collision_reward': 'NON_COLLISION_REW',
                'total_right_lane_reward': 'RIGHT_LANE_REW',
                'total_high_speed_reward': 'HIGH_SPEED_REW',
                'total_low_speed_reward': 'LOW_SPEED_REW',
               }

RAINBOW_TRAJ_STATICS = {'average': 'AVG',
                        'std': 'STD',
                        'median': 'MEDIAN',
                        'min': 'MIN',
                        'max': 'MAX',
                        }

RAINBOW_CF_TRAJ_METRICS = {
                'true_episode_reward': 'TR',
                'true_raw_episode_reward': 'TRR',
                'episode_length': 'LEN',
               }

RAINBOW_CF_TRAJ_STATICS = {'average': 'AVG',
                        'std': 'STD',
                        'median': 'MEDIAN',
                        'min': 'MIN',
                        'max': 'MAX',
                        }

AGENT_EVAL_FORMAT = {
    'sac':[
        ('episode', 'E', 'int'),
        ('step', 'S', 'int'),
        ('episode_length', 'LEN', 'float'),
        # ('episode_reward', 'R', 'float'),
        ('episode_r_hat', 'RH', 'float'),
        ('true_episode_reward_avg', 'TR_AVG', 'float'),
        ('true_episode_reward_std', 'TR_STD', 'float'),
        ('true_episode_reward_median', 'TR_MED', 'float'),
        ('true_episode_reward_min', 'TR_MIN', 'float'),
        ('true_episode_reward_max', 'TR_MAX', 'float'),
        ('true_episode_success', 'TS', 'float'),
        ('duration', 'D', 'time'),
    ],
    'rainbow': [
        ('step', 'S', 'int'),
        ('iter', 'ITR', 'float'),  # because for ftRL, we evaluate the agent between ft and ftRL
        ('duration', 'D', 'time'),
        ('eps_this_eval', 'EPS', 'float'),
        ('model_save_duration', 'SAVE_D', 'time'),
        # Items in RAINBOW_TRAJ_METRICS will be added in logger.__init__
    ],
    'SL': [
        ('step', 'S', 'int'),
        ('duration', 'D', 'time'),
        ('eps_this_eval', 'EPS', 'float'),
        ('model_save_duration', 'SAVE_D', 'time'),
        ('acc_thres', 'ACC_THRES', 'float'),
        ('human_acc', 'ACC_HUMAN', 'float'),
        ('sl_epoch', 'EPO', 'int'),
        # Items in RAINBOW_TRAJ_METRICS will be added in logger.__init__
    ]
}


AGENT_TRAIN_FORMAT = {
    'sac': [
        ('episode', 'E', 'int'),
        ('step', 'S', 'int'),
        ('episode_length', 'LEN', 'float'),
        # ('episode_reward', 'R', 'float'),
        ('episode_r_hat', 'RH', 'float'),
        ('true_episode_reward', 'TR', 'float'), 
        ('total_feedback', 'TF', 'int'),
        ('labeled_feedback', 'LR', 'int'),
        ('noisy_feedback', 'NR', 'int'),
        ('duration', 'D', 'time'),
        ('total_duration', 'TD', 'time'),

        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('bc_loss', 'BCLOSS', 'float'),
    ],
    'sac_discrete': [
        ('episode', 'E', 'int'),
        ('step', 'S', 'int'),
        ('episode_length', 'LEN', 'float'),
        # ('episode_reward', 'R', 'float'),
        ('episode_r_hat', 'RH', 'float'),
        ('true_episode_reward', 'TR', 'float'), 
        ('total_feedback', 'TF', 'int'),
        ('labeled_feedback', 'LR', 'int'),
        ('noisy_feedback', 'NR', 'int'),
        ('duration', 'D', 'time'),
        ('total_duration', 'TD', 'time'),

        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('bc_loss', 'BCLOSS', 'float'),
    ],
    'rainbow': [
        ('step', 'S', 'int'),
        ('iter', 'ITR', 'int'),
        ('duration', 'D', 'time'),
        ('total_duration', 'TD', 'time'),

        ('loss', 'RL_L', 'float'),
        ('gradNorm', 'RL_GN', 'float'),
        ('tdAbsErr', 'RL_TD_ERR', 'float'),
    ],
    'SL': [],
}

SL_EPO_FORMAT = {
    'SL': [
            ('epoch', 'EPO', 'int'),
            ('human_acc', 'ACC', 'float'),
            ('loss', 'LOSS', 'float'),
            ('grad_norm', 'GN', 'float'),
    ]
}

INTERACT_FORMAT = {
    'GT': [],
    'PEBBLEAtari': [
        ('step', 'S', 'int'),
        ('feedback_count', 'FC', 'int'),
        # ('reward_acc', 'RACC', 'float'),
        ('feedback_this_itr', 'FC_ITR', 'int'),
        ('label_count', 'LC', 'int'),
        ('label_this_itr', 'LC_ITR', 'int'),
        ('len_inputs_T', 'LEN_IN_T', 'int'),
        ('len_label', 'LEN_LAB', 'int'),
        ('reward_update', 'RU', 'int'),
        ('final_action_acc', 'AACC', 'float'),
        ('final_action_acc_max', 'AACC_MAX', 'float'),
        ('final_loss', 'LOSS', 'float'),
        ('final_loss_min', 'LOSS_MIN', 'float'),
        ('learn_duration', 'LD', 'time'),
        ('sample_duration', 'SD', 'time'),
        ('relabel_duration', 'RLD', 'time'),
    ],
    'CF_ft': [
        ('step', 'S', 'int'),
        ('total_segment', 'SEGC', 'int'),
        ('segment_this_itr', 'SEGC_ITR', 'int'),
        ('total_label', 'LABC', 'int'),
        ('label_this_itr', 'LABC_ITR', 'int'),
        ('len_inputs_T', 'LEN_IN_T', 'int'),
        ('len_label', 'LEN_LAB', 'int'),
        ('reward_update', 'RU', 'int'),
        ('final_action_acc', 'AACC', 'float'),
        ('final_loss', 'LOSS', 'float'),
        ('final_loss_margine', 'LOSS_MAR', 'float'),
        ('final_LR_acc', 'LR_ACC', 'float'),
        ('final_loss_LR', 'LOSS_LR', 'float'),
        ('loss_margine', 'L_MARG', 'float'),
        ('finetune_duration', 'FD', 'time'),
        ('sample_duration', 'SD', 'time'),
        ('q_diff_average', 'QDiffAvg', 'float'),
    ],
    'CF_RLft': [
        # same items as CF_ft
        ('step', 'S', 'int'),
        ('iter', 'ITR', 'float'),
        ('sample_eps', 'EPS', 'float'),
        ('total_segment', 'SEGC', 'int'),
        ('segment_this_itr', 'SEGC_ITR', 'int'),
        ('total_label', 'LABC', 'int'),
        ('label_this_itr', 'LABC_ITR', 'int'),
        ('len_inputs_T', 'LEN_IN_T', 'int'),
        ('len_label', 'LEN_LAB', 'int'),
        ('total_update', 'UPDATE', 'int'),
        ('ft_human_train_bs', 'HUMAN_BS', 'int'),
        ('ft_tgt_train_bs', 'TGT_BS', 'int'),
        ('final_human_acc', 'HUMAN_ACC', 'float'),
        ('final_loss', 'LOSS', 'float'),
        # ('final_loss_margine', 'LOSS_MAR', 'float'),
        # ('loss_margine', 'L_MARG', 'float'),
        ('finetune_duration', 'FD', 'time'),
        ('sample_duration', 'SD', 'time'),
        ('q_diff_average', 'QDiffAvg', 'float'),
        ('unique_CF_labels', 'UniqueCF', 'int'),
        # ('tgt_test_acc', 'TGT_TEST_ACC', 'float'),
        # ('final_tgt_act_acc', 'TGT_AACC', 'float'),
        ('tgt_oracle_act_acc_avg', 'TGT_ORACLE_ACC_AVG', 'float'),
        ('agent_match_tgt_acc_avg', 'AGENT_TGT_ACC_AVG', 'float'),
        ('agent_match_oracle_acc_avg', 'AGENT_ORACLE_ACC_AVG', 'float'),
        ('final_tgt_losses', 'TGT_LOSS', 'float'),
        ('final_human_losses', 'HUMAN_LOSS', 'float'),
        ('ft_use_tgt', 'USE_TGT', 'bool'),
        ('ft_w_human_avg', 'W_HUMAN_AVG', 'float'),
        ('ft_w_tgt_avg', 'W_HUMAN_AVG', 'float'),
    ],
    'CF': [
        ('step', 'S', 'int'),
        ('feedback_count', 'FC', 'int'),
        ('feedback_this_itr', 'FC_ITR', 'int'),
        ('label_count', 'LC', 'int'),
        ('label_this_itr', 'LC_ITR', 'int'),
        ('len_inputs_T', 'LEN_IN_T', 'int'),
        ('len_label', 'LEN_LAB', 'int'),
        ('reward_update', 'RU', 'int'),
        ('final_action_acc', 'AACC', 'float'),
        ('final_action_acc_max', 'AACC_MAX', 'float'),
        ('final_loss', 'LOSS', 'float'),
        ('final_loss_min', 'LOSS_MIN', 'float'),
        ('learn_duration', 'LD', 'time'),
        ('sample_duration', 'SD', 'time'),
        ('relabel_duration', 'RLD', 'time'),
    ],
    'SL': [],
}

RLLOSS_FORMAT = {
    'CF_RLft':[
        # extra special items
        ('epoch_cnt', 'RLEPO', 'int'),
        ('CF_agent_pred_human_acc', 'HUMAN_ACC', 'float'),
        ('CF_agent_pred_tgt_acc', 'TGT_ACC', 'float'),
        ('CF_agent_pred_RND_acc', 'RND_ACC', 'float'),
        ('rl_1_loss_avg', 'RL1LOSS', 'float'),
        ('rl_n_loss_avg', 'RLnLOSS', 'float'),
        ('sl_human_loss_avg', 'HUMAN_SLOSS', 'float'),
        ('sl_tgt_loss_avg', 'TGT_SLLOSS', 'float'),
        ('sl_RND_loss_avg', 'RND_SLLOSS', 'float'),
        ('loss_avg', 'ALL_LOSS', 'float'),
        ('grad_norms_avg', 'GN', 'float'),
        ('tgt_RND_confident_ratio_avg', 'TGT_RNDR_AVG', 'float'),
        ('wrong_doubt_ratio_avg', 'TGT_RND_WRONGR_AVG', 'float'),
    ],
}

INTERACT_ACC_FORMAT = {
    'GT': [],
    'CF': [
        ('step', 'S', 'int'),
        ('epoch', 'EPO', 'int'),
        # ('id_ensemble', 'ID_EN', 'int'),
        ('action_acc_ensemble_0', 'AACC_0', 'float'),
        ('action_acc_ensemble_1', 'AACC_1', 'float'),
        ('action_acc_ensemble_2', 'AACC_2', 'float'),
        ('action_acc_avg', 'AACC_AVG', 'float'),
        ('action_acc_max', 'AACC_MAX', 'float'),
        ('action_acc_min', 'AACC_MIN', 'float'),
        ('action_acc_std', 'AACC_STD', 'float'),
        ('loss_ensemble_0', 'LOSS_0', 'float'),
        ('loss_ensemble_1', 'LOSS_1', 'float'),
        ('loss_ensemble_2', 'LOSS_2', 'float'),
        ('loss_avg', 'LOSS_AVG', 'float'),
        ('loss_max', 'LOSS_MAX', 'float'),
        ('loss_min', 'LOSS_MIN', 'float'),
        ('loss_std', 'LOSS_STD', 'float'),
        ('grad_norm_ensemble_0', 'GN_0', 'float'),
        ('grad_norm_ensemble_1', 'GN_1', 'float'),
        ('grad_norm_ensemble_2', 'GN_2', 'float'),
    ],
    'CF_ft': [
        ('step', 'S', 'int'),
        ('epoch', 'EPO', 'int'),
        # ('id_ensemble', 'ID_EN', 'int'),
        ('acc', 'ACC', 'float'),
        ('loss', 'LOSS', 'float'),
        ('LR_acc', 'LR_ACC', 'float'),
        ('loss_margine', 'LOSS_MAR', 'float'),
        ('loss_LR', 'LOSS_LR', 'float'),
        ('grad_norm', 'GN', 'float'),
    ],
    'CF_RLft': [
        # same with CF_ft
        ('step', 'S', 'int'),
        ('epoch', 'EPO', 'int'),
        ('human_acc', 'HUMAN_ACC', 'float'),
        ('loss', 'LOSS', 'float'),
        # ('loss_margine', 'LOSS_MAR', 'float'),
        ('grad_norm', 'GN', 'float'),
        ('tgt_oracle_act_acc', 'TGT_ORACLE_ACC', 'float'),
        ('agent_match_tgt_acc', 'AGENT_TGT_ACC', 'float'),
        ('agent_match_oracle_acc', 'AGENT_ORACLE_ACC', 'float'),
        ('tgt_losses', 'TGT_LOSS', 'float'),
        ('human_losses', 'HUMAN_LOSS', 'float'),
        ('ft_use_tgt', 'USE_TGT', 'bool'),
        ('ft_w_human', 'W_HUMAN', 'float'),
        ('ft_w_tgt', 'W_TGT', 'float'),
    ],
    'PEBBLEAtari': [  # same as 'CF'
        ('step', 'S', 'int'),
        ('epoch', 'EPO', 'int'),
        # ('id_ensemble', 'ID_EN', 'int'),
        ('action_acc_ensemble_0', 'AACC_0', 'float'),
        ('action_acc_ensemble_1', 'AACC_1', 'float'),
        ('action_acc_ensemble_2', 'AACC_2', 'float'),
        ('action_acc_avg', 'AACC_AVG', 'float'),
        ('action_acc_max', 'AACC_MAX', 'float'),
        ('action_acc_min', 'AACC_MIN', 'float'),
        ('action_acc_std', 'AACC_STD', 'float'),
        ('loss_ensemble_0', 'LOSS_0', 'float'),
        ('loss_ensemble_1', 'LOSS_1', 'float'),
        ('loss_ensemble_2', 'LOSS_2', 'float'),
        ('loss_avg', 'LOSS_AVG', 'float'),
        ('loss_max', 'LOSS_MAX', 'float'),
        ('loss_min', 'LOSS_MIN', 'float'),
        ('loss_std', 'LOSS_STD', 'float'),
        ('grad_norm_ensemble_0', 'GN_0', 'float'),
        ('grad_norm_ensemble_1', 'GN_1', 'float'),
        ('grad_norm_ensemble_2', 'GN_2', 'float'),
    ],
    'SL': [],
}

RLLOSS_EPOCH_FORMAT = {
    'CF_RLft': [
        # special items for RLft
        ('epoch', 'RLEPO', 'int'),
        ('CF_agent_pred_human_acc', 'HUMAN_ACC', 'float'),
        ('CF_agent_pred_tgt_acc', 'TGT_ACC', 'float'),
        ('CF_agent_pred_RND_acc', 'RND_ACC', 'float'),
        ('rl_1_loss_avg', 'RL1LOSS', 'float'),
        ('rl_n_loss_avg', 'RLnLOSS', 'float'),
        ('sl_human_loss_avg', 'HUMAN_SLOSS', 'float'),
        ('sl_tgt_loss_avg', 'TGT_SLLOSS', 'float'),
        ('sl_RND_loss_avg', 'RND_SLLOSS', 'float'),
        ('loss_avg', 'ALL_LOSS', 'float'),
        ('grad_norms_avg', 'GN', 'float'),
        ('tgt_RND_confident_ratio', 'TGT_RND_RAT', 'float'),
        ('wrong_doubt_ratio', 'TGT_RND_WRONG_RAT', 'float'),
    ],
}

COLOR = {
    'train': 'yellow',
    'eval': 'green',
    'interact': 'blue',
    'interact_acc': 'light_blue',
    'rlloss_epoch': 'magenta',
    'rlloss': 'light_magenta',
    'SL_epoch': 'light_magenta',
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1, repeat_count=1):
        if value is not None:
            self._sum += value
            self._count += n
        assert self._count <= repeat_count
        # if self._count > 1:
        #     pdb.set_trace()   # Normally, for rainbow, shouldn't have more tan one data logged at the same time?

    def value(self):
        # If a initialized AverageMeter has never been added with data, it will return None instead of 0
        return self._sum / max(1, self._count) if self._count > 0 else None


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1, repeat_count=1):
        self._meters[key].update(value, n, repeat_count=repeat_count)  # accumulative sum (will be averaged in dump())

    def _prime_meters(self, prefix):
        data = dict()
        assert prefix in ['train', 'eval', 'interact', 'interact_acc',\
                          'rlloss', 'rlloss_epoch', 'SL_epoch']
        for key, meter in self._meters.items():
            assert key[0] == prefix[0]  # make sure that current logged data only contains related info
            key = key[len(prefix)+1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            return f'{key}: ' + (f'{int(value)}' if value is not None else 'None')
        elif ty == 'float':
            return f'{key}: ' + (f'{value:.07f}' if value is not None else 'None')
        elif ty == 'time':
            return f'{key}: ' + (f'{value:.1f}' if value is not None else 'None')
        elif ty == 'bool':
            return f'{key}: ' + (f'{bool(value)}' if value is not None else 'None')
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, COLOR[prefix])
        pieces = [f'| {prefix: <9}']  # set number of the reserved blank spaces
        for key, disp_key, ty in self._formating:
            value = data.get(key, None)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            print(f'[WARNING]: no log but dump it.')
            return
        if save:
            data = self._prime_meters(prefix=prefix)
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=False,
                #  log_frequency=10000,
                 agent='sac',
                 reward_type='GT',
                 traj_based=False,
                 env_name=None):
        self._log_dir = log_dir
        # self._log_frequency = log_frequency
        if save_tb:  # tb means tensorboard
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        # each agent has specific output format for training
        if env_name == 'atari':
            for metric_name, static_name in itertools.product(
                        ATARI_TRAJ_METRICS, RAINBOW_TRAJ_STATICS):
                AGENT_EVAL_FORMAT['rainbow'].append(
                    (f'{metric_name}_{static_name}',
                    f'{ATARI_TRAJ_METRICS[metric_name]}_{RAINBOW_TRAJ_STATICS[static_name]}',
                    'float')
                )
                AGENT_EVAL_FORMAT['SL'].append(
                    (f'{metric_name}_{static_name}',
                    f'{ATARI_TRAJ_METRICS[metric_name]}_{RAINBOW_TRAJ_STATICS[static_name]}',
                    'float')
                )
        elif env_name == 'highway':
            for metric_name, static_name in itertools.product(
                        HIGHWAY_TRAJ_METRICS, RAINBOW_TRAJ_STATICS):
                AGENT_EVAL_FORMAT['rainbow'].append(
                    (f'{metric_name}_{static_name}',
                    f'{HIGHWAY_TRAJ_METRICS[metric_name]}_{RAINBOW_TRAJ_STATICS[static_name]}',
                    'float')
                )
                AGENT_EVAL_FORMAT['SL'].append(
                    (f'{metric_name}_{static_name}',
                    f'{HIGHWAY_TRAJ_METRICS[metric_name]}_{RAINBOW_TRAJ_STATICS[static_name]}',
                    'float')
                )
        else:
            raise NotImplementedError
        assert agent in AGENT_TRAIN_FORMAT
        assert agent in AGENT_EVAL_FORMAT
        assert reward_type in INTERACT_FORMAT
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=AGENT_TRAIN_FORMAT[agent])
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=AGENT_EVAL_FORMAT[agent])
        if reward_type != 'GT' and traj_based:
            for metric_name, static_name in itertools.product(
                        RAINBOW_CF_TRAJ_METRICS, RAINBOW_CF_TRAJ_STATICS):
                INTERACT_FORMAT[reward_type].append(
                    (f'{metric_name}_{static_name}',
                    f'{RAINBOW_CF_TRAJ_METRICS[metric_name]}_{RAINBOW_CF_TRAJ_STATICS[static_name]}',
                    'float')
                )
            INTERACT_FORMAT[reward_type].extend(
                [('traj_this_itr', 'NUM_TRAJ', 'int'),
                ('total_traj', 'TOTAL_TRAJ', 'int'),
                ('feedback_traj_avg', 'FC_TRAJ', 'float')]
            )
        self._interact_mg = MetersGroup(os.path.join(log_dir, 'interact'),
                                        formating=INTERACT_FORMAT[reward_type])
        self._interact_acc_mg = MetersGroup(os.path.join(log_dir, 'interact_acc'),
                                        formating=INTERACT_ACC_FORMAT[reward_type])
        if reward_type in RLLOSS_FORMAT:
            self._rlloss_mg = MetersGroup(os.path.join(log_dir, 'rlloss'),
                                        formating=RLLOSS_FORMAT[reward_type])
        else:
            self._rlloss_mg = None
        if reward_type in RLLOSS_EPOCH_FORMAT:
            self._rlloss_epoch_mg = MetersGroup(os.path.join(log_dir, 'rlloss_epoch'),
                                        formating=RLLOSS_EPOCH_FORMAT[reward_type])
        else:
            self._rlloss_epoch_mg = None
        if reward_type in SL_EPO_FORMAT:
            self._sl_epo_mg = MetersGroup(os.path.join(log_dir, 'SL_epo'),
                                        formating=SL_EPO_FORMAT[reward_type])
        else:
            self._sl_epo_mg = None

    def _should_log(self, step, log_frequency):
        # log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, n, step):
        if self._sw is not None and value is not None:
            self._sw.add_scalar(key, value / n, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1, log_frequency=1,
            repeat_count=1):
        if not self._should_log(step, log_frequency):
            return
        # assert key.startswith('train') or key.startswith('eval') or key.startswith('interact')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, n, step)
        if key.startswith('train'):
            mg = self._train_mg 
        elif key.startswith('eval'):
            mg = self._eval_mg
        elif key.startswith('interact_acc'):
            mg = self._interact_acc_mg
        elif key.startswith('interact'):
            mg = self._interact_mg
        elif key.startswith('rlloss_epoch'):
            mg = self._rlloss_epoch_mg
        elif key.startswith('rlloss'):
            mg = self._rlloss_mg
        elif key.startswith('SL_epoch'):
            mg = self._sl_epo_mg
        else:
            raise NotImplementedError
        mg.log(key, value, n, repeat_count=repeat_count)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval') or key.startswith('interact')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval') or key.startswith('interact')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
            self._interact_acc_mg.dump(step, 'interact_acc', save)
            self._interact_mg.dump(step, 'interact', save)
            if self._rlloss_mg is not None:
                self._rlloss_mg.dump(step, 'rlloss', save)
            if self._rlloss_epoch_mg is not None:
                self._rlloss_epoch_mg.dump(step, 'rlloss_epoch', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        elif ty == 'interact_acc':
            self._interact_acc_mg.dump(step, 'interact_acc', save)
        elif ty == 'interact':
            self._interact_mg.dump(step, 'interact', save)
        elif ty == 'rlloss':
            self._rlloss_mg.dump(step, 'rlloss', save)
        elif ty == 'rlloss_epoch':
            self._rlloss_epoch_mg.dump(step, 'rlloss_epoch', save)
        elif ty == 'SL_epoch':
            self._sl_epo_mg.dump(step, 'SL_epoch', save)
        else:
            raise f'invalid log type: {ty}'
