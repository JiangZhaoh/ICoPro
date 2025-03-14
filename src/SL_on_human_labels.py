import os
import time
import pickle as pkl
import pdb
import torch
import random
import numpy as np
import hydra
import itertools
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from old_utils.utils import set_seed_everywhere
from old_utils.logger import Logger, RAINBOW_TRAJ_STATICS, \
                                ATARI_TRAJ_METRICS, HIGHWAY_TRAJ_METRICS
from new_utils.rlpyt_utils import GymAtariTrajInfo, GymHighwayTrajInfo
from new_utils import atari_env, highway_env
from new_utils import atari_env
from finetune_RLRND_rainbow_Q import Workspace

class SLWorkspace(Workspace):
    def __init__(self, SL_cfg):  # Based on build_and_train() + MinibatchRlEvalWandb.train().startup()
        data_parent_path = ''
        for file_pre_t, var_t in zip([SL_cfg.data_path.file_pre_1,
                                        SL_cfg.data_path.file_pre_2,
                                        SL_cfg.data_path.file_pre_3],
                                     [SL_cfg.data_path.var_1,
                                        SL_cfg.data_path.var_2,
                                        SL_cfg.data_path.var_3]):
            if file_pre_t is None:
                break
            else:
                data_parent_path += f'{file_pre_t}{var_t}'
        data_parent_path += SL_cfg.data_path.file_end
        data_path = os.path.join(SL_cfg.data_path.root, data_parent_path, 'exp_logs')
        cfg = OmegaConf.create(
                DictConfig(pkl.load(open(f'{data_path}/metadata.pkl', 'rb'))).cfg)
        cfg.seed = SL_cfg.seed
        cfg.device = SL_cfg.device
        cfg.finetune_cfg.model_update = SL_cfg.model_update
        cfg.finetune_cfg.acc_target = SL_cfg.acc_target
        if SL_cfg.num_eval_episodes is not None:
            cfg.num_eval_episodes = SL_cfg.num_eval_episodes
        cfg.finetune_cfg.finetune_model.oracle_type='oq'

        label_buffer_statis = pkl.load(open(f'{data_path}/label_buffer_statis.pkl', 'rb'))
        if SL_cfg.eval_itr == -1:
            statis_ls_idx = -1
            len_buffer = label_buffer_statis['len_label'][-1]
        else:
            statis_ls_idx = None
            for statis_ls_idx, itr in enumerate(label_buffer_statis['itr']):
                if itr == SL_cfg.eval_itr:
                    break
            len_buffer = label_buffer_statis['len_label'][statis_ls_idx]
        
        label_buffer_list = [fn for fn in os.listdir(data_path) \
                                    if fn.endswith('.pkl') and\
                                        fn.startswith('label_buffer_') and\
                                        fn!='label_buffer_statis.pkl']
        assert len(label_buffer_list) == 1
        label_buffer = pkl.load(open(f'{data_path}/{label_buffer_list[0]}', 'rb'))
        label_buffer = {k: v[:len_buffer] for k, v in label_buffer.items()}
        
        # device
        assert cfg.device in ['cuda', 'cpu']
        cfg.device = 'cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu'
        print(f"***** device is {cfg.device} *****")
        if cfg.device == 'cuda':
            print(f"***** torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)} *****")

        self.work_dir = os.path.abspath(HydraConfig.get().run.dir)
        print(f'workspace: {self.work_dir}')

        if cfg.env.env_name.lower() in atari_env.ATARI_ENV:
            self.is_atari = True
            self.is_highway = False
            self.TrajInfoCls = GymAtariTrajInfo

            self.eval_envs = [atari_env.make_env(cfg, eval=True) for _ in range(cfg.num_eval_episodes)]
            for id_env in range(cfg.num_eval_episodes):
                self.eval_envs[id_env].reset(seed=cfg.seed+id_env+123321778)
            self.total_env_lives = self.eval_envs[0].ale.lives()
            
            self.action_names = self.eval_envs[0].unwrapped.get_action_meanings()

            self.logger = Logger(self.work_dir,
                                save_tb=False,
                                agent='SL',
                                reward_type='SL',
                                traj_based=False,
                                env_name='atari')
        elif cfg.env.env_name in highway_env.HIGHWAY_ENV_NAME:
            self.is_atari = False
            self.is_highway = True
            self.TrajInfoCls = GymHighwayTrajInfo

            eval_env_cfg = copy.deepcopy(cfg)
            if cfg.save_eval_video.flag:
                eval_env_cfg.env.render_mode = 'rgb_array'
            self.eval_envs = [highway_env.make_highway_env(eval_env_cfg, eval=True) for _ in range(cfg.num_eval_episodes)]
            for id_env in range(cfg.num_eval_episodes):
                self.eval_envs[id_env].reset(seed=cfg.seed+id_env+123321778)
            self.total_env_lives = 1
            
            self.action_names = self.eval_envs[0].action_type.actions
            
            self.logger = Logger(self.work_dir,
                                save_tb=False,
                                agent='SL',
                                reward_type='SL',
                                traj_based=False,
                                env_name='highway')
        else:
            raise NotImplementedError

        if self.is_atari:
            self.evaluate_agent = self.evaluate_agent_atari
        elif self.is_highway:
            self.evaluate_agent = self.evaluate_agent_highway
        else:
            raise NotImplementedError

        set_seed_everywhere(cfg.seed)  # NOTE: this part doesn't set seed for env
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(cfg.device)

        self.rank = 0  # from runner.startup()
        self.world_size = 1  # from runner.startup()
        self.global_B = cfg.sampler_b
        self.env_ranks = list(range(0, cfg.sampler_b))
        
        # NOTE: we do not write SerialSampler explicitly, in order to make the code more consistent with PEBBLE
        self.agent = hydra.utils.instantiate(cfg.agent.agent)
        # NOTE: exactly the same configuration as the old agent who collect the label buffer,
        #       including initial agent ''ckpt_path'' !
        self.agent.initialize(
                              action_dim=cfg.agent.agent.model.action_dim,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
        
        if cfg.agent.agent.model.noisy:
            # TODO: I'm not sure if we should adding noise when using noisy-net for fine-tuning
            assert (cfg.finetune_cfg.noise_override is None) or\
                  (cfg.finetune_cfg.noise_override == False)
            # Note: if noise_override is None: finetuning = use noisy net, eval = no noisy net
            #                         if False:  both finetuning and eval will not use noisy net
            self.agent.model.head.set_noise_override(cfg.finetune_cfg.noise_override)
            self.agent.target_model.head.set_noise_override(cfg.finetune_cfg.noise_override)
        
        self.agent.to_device(self.device)
        self.agent.give_V_min_max(-cfg.agent.algo.V_max,
                                  cfg.agent.algo.V_max)
        
        self.finetune_model = hydra.utils.instantiate(cfg.finetune_cfg.finetune_model)
        self.finetune_model.config_agent(agent=self.agent)  # will config agent & optimizer for that agent's parameters here
        
        self.finetune_model.reset_label_buffer(label_buffer)

        self.iterative_misspelling_check(cfg, token='Fasle')  # have mis-spell this config many times... 
        self.iterative_misspelling_check(SL_cfg, token='Fasle')  # have mis-spell this config many times... 
        
        if SL_cfg.eval_thres == True:
            SL_cfg.eval_acc_thres_ls = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9,
                                    0.925, 0.95, 0.975, 0.98, 0.99, 0.999]
        else:
            SL_cfg.eval_acc_thres_ls = [1.1]
        
        self.eval_acc_thres_ls = np.array(SL_cfg.eval_acc_thres_ls)
        
        self.cfg = cfg
        self.SL_cfg = SL_cfg
        os.makedirs(self.work_dir, exist_ok=True)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': cfg, 'SL_cfg': SL_cfg}, open(meta_file, "wb"))

    @torch.no_grad()
    def evaluate_agent_atari(self, itr, agent=None, eval_eps=None):
        eval_begin = time.time()
        eval_completed_traj_infos, eps_this_eval = self.collect_evaluation_atari(itr, agent,
                                                        save_info=self.cfg.save_eval_img,
                                                        eval_eps=eval_eps)
        eval_end = time.time()
        eval_duration = eval_end - eval_begin
        if len(eval_completed_traj_infos) == 0:
            print("!!!!!WARNING: had no complete trajectories in eval at iter {itr}.!!!!!")
    
        self.logger.log('eval/duration', eval_duration, self.sl_epoch)
        if agent is None:
            self._cum_eval_time += eval_duration
        
        self.logger.log('eval/num_completed_traj', len(eval_completed_traj_infos), self.sl_epoch)

        true_episode_reward_list = [info["Return"] for info in eval_completed_traj_infos]
        episode_reward_hat_list = [info["ReturnRHat"] for info in eval_completed_traj_infos]
        true_raw_episode_reward_list = [info["RawReturn"] for info in eval_completed_traj_infos]
        episode_length_list = [info["Length"] for info in eval_completed_traj_infos]

        for lst_name, stat_name in itertools.product(
                        ATARI_TRAJ_METRICS.keys(), RAINBOW_TRAJ_STATICS.keys()):
            self.logger.log(f'eval/{lst_name}_{stat_name}',
                             eval(f'np.{stat_name}({lst_name}_list)'), self.sl_epoch)
        
        self.logger.log('eval/eps_this_eval', eps_this_eval, self.sl_epoch)
        self.logger.log('eval/iter', itr, self.sl_epoch)
        self.logger.dump(self.sl_epoch, ty='eval')

    @torch.no_grad()
    def evaluate_agent_highway(self, itr, agent=None, eval_eps=None):
        eval_begin = time.time()
        eval_completed_traj_infos, eps_this_eval = self.collect_evaluation_highway(itr, agent,
                                                        save_info=self.cfg.save_eval_img,
                                                        eval_eps=eval_eps)
        eval_end = time.time()
        eval_duration = eval_end - eval_begin
        if len(eval_completed_traj_infos) == 0:
            print("!!!!!WARNING: had no complete trajectories in eval at iter {itr}.!!!!!")
    
        self.logger.log('eval/duration', eval_duration, self.sl_epoch)
        if agent is None:
            self._cum_eval_time += eval_duration
        
        # steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        # logger.record_tabular('StepsInEval', steps_in_eval)
        self.logger.log('eval/num_completed_traj', len(eval_completed_traj_infos), self.sl_epoch)
        total_reward_list = [info["total_reward"] for info in eval_completed_traj_infos]
        cnt_step_list = [info["cnt_step"] for info in eval_completed_traj_infos]
        average_speed_list = [info["average_speed"] for info in eval_completed_traj_infos]
        average_forward_speed_list = [info["average_forward_speed"] for info in eval_completed_traj_infos]
        crashed_list = [info["Episodic_info"]["crashed"] for info in eval_completed_traj_infos]
        truncated_list = [info["Episodic_info"]["truncated"] for info in eval_completed_traj_infos]
        total_time_list = [info["Episodic_info"]["time"] for info in eval_completed_traj_infos]
        total_distance_list = [info["Episodic_info"]["total_distance"] for info in eval_completed_traj_infos]
        cnt_lane_changed_list = [info["Episodic_info"]["cnt_lane_changed"] for info in eval_completed_traj_infos]
        total_off_road_time_list = [info["Episodic_info"]["total_off_road_time"] for info in eval_completed_traj_infos]
        total_collision_reward_list = [info["Episodic_info"]["total_collision_reward"] for info in eval_completed_traj_infos]
        total_non_collision_reward_list = [info["Episodic_info"]["total_non_collision_reward"] for info in eval_completed_traj_infos]
        total_right_lane_reward_list = [info["Episodic_info"]["total_right_lane_reward"] for info in eval_completed_traj_infos]
        total_high_speed_reward_list = [info["Episodic_info"]["total_high_speed_reward"] for info in eval_completed_traj_infos]
        total_low_speed_reward_list = [info["Episodic_info"]["total_low_speed_reward"] for info in eval_completed_traj_infos]
        # total_lane_change_reward_list = [info["Episodic_info"]["total_lane_change_reward"] for info in eval_completed_traj_infos]
        for lst_name, stat_name in itertools.product(
                        HIGHWAY_TRAJ_METRICS.keys(), RAINBOW_TRAJ_STATICS.keys()):
            self.logger.log(f'eval/{lst_name}_{stat_name}',
                             eval(f'np.{stat_name}({lst_name}_list)'), self.sl_epoch)
        
        self.logger.log('eval/eps_this_eval', eps_this_eval, self.sl_epoch)
        self.logger.log('eval/iter', itr, self.sl_epoch)
        self.logger.dump(self.sl_epoch, ty='eval')

    def SL_on_human_labels(self):
        self._cum_eval_time = 0
        # NOTE: be careful: explicitly change the ft_use_tgt flag
        self.finetune_model.ft_use_tgt = False
        self.logger.log('eval/acc_thres', 0, 0)
        self.logger.log('eval/human_acc', 0, 0)
        self.logger.log('eval/sl_epoch', 0, 0)
        self.sl_epoch = 0
        self.evaluate_agent(itr=0,
                            # epoch=0,
                            eval_eps=self.cfg.agent.agent.eps_eval \
                                if self.cfg.agent.agent.ckpt_path is not None else 1.)
        
        for sl_epoch in range(1, self.SL_cfg.model_update):  # epoch count starting from 1 to avoid tgt_in_ft settings in finetune_model.finetune()
            self.sl_epoch = sl_epoch
            ft_train_human_acc, ft_train_losses, ft_train_grad_norms,\
                _, _, _,_, _ = self.finetune_model.finetune(itr=-1,
                                                            ft_epoch=sl_epoch)
            
            self.logger.log('SL_epoch/epoch', sl_epoch, sl_epoch)
            self.logger.log('SL_epoch/human_acc', ft_train_human_acc, sl_epoch)
            self.logger.log('SL_epoch/loss', ft_train_losses, sl_epoch)
            self.logger.log('SL_epoch/grad_norm', ft_train_grad_norms, sl_epoch)
            self.logger.dump(sl_epoch, ty='SL_epoch')

            evaled = False
            if sl_epoch == 1:
                st_acc_thres_idx = np.sum(self.eval_acc_thres_ls < ft_train_human_acc)
            if (st_acc_thres_idx < len(self.eval_acc_thres_ls)) and\
                ft_train_human_acc >= self.eval_acc_thres_ls[st_acc_thres_idx]:
                self.logger.log('eval/acc_thres', self.eval_acc_thres_ls[st_acc_thres_idx], sl_epoch)
                self.logger.log('eval/human_acc', ft_train_human_acc, sl_epoch)
                self.logger.log('eval/sl_epoch', sl_epoch, sl_epoch)
                self.evaluate_agent(itr=1,
                                # epoch=sl_epoch,
                                eval_eps=self.cfg.agent.agent.eps_eval)
                st_acc_thres_idx += 1
                evaled = True
            
            if ft_train_human_acc >= self.SL_cfg.acc_target and\
                st_acc_thres_idx == len(self.eval_acc_thres_ls):
                break
        
        if not evaled:
            self.logger.log('eval/acc_thres', ft_train_human_acc, sl_epoch)
            self.logger.log('eval/human_acc', ft_train_human_acc, sl_epoch)
            self.logger.log('eval/sl_epoch', sl_epoch, sl_epoch)
            self.evaluate_agent(itr=1,
                            # epoch=sl_epoch,
                            eval_eps=self.cfg.agent.agent.eps_eval)

@hydra.main(version_base=None, config_path="config")
def main(SL_cfg: DictConfig):
    workspace = SLWorkspace(SL_cfg)
    workspace.SL_on_human_labels()

if __name__ == '__main__':
    main()