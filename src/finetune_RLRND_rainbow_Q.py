#!/usr/bin/env python3
import os
import time
import pickle as pkl
import pdb
import torch
import numpy as np
import hydra
import itertools
import copy
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from rlpyt.samplers.collections import BatchSpec
from rlpyt.utils.buffer import buffer_from_example

from old_utils.utils import set_seed_everywhere
from old_utils.logger import Logger, RAINBOW_TRAJ_STATICS,\
                                ATARI_TRAJ_METRICS, HIGHWAY_TRAJ_METRICS
from new_utils.rlpyt_utils import delete_ind_from_array, build_samples_buffer,\
                                    GymAtariTrajInfo, GymHighwayTrajInfo
from new_utils import atari_env, highway_env
from new_utils.tensor_utils import  torchify_buffer, numpify_buffer
from new_utils.draw_utils import ax_plot_img, ax_plot_bar, ax_plot_heatmap


class Workspace(object):
    def __init__(self, cfg):  # Based on build_and_train() + MinibatchRlEvalWandb.train().startup()
        # device
        assert cfg.device in ['cuda', 'cpu']
        cfg.device = 'cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu'
        print(f"***** device is {cfg.device} *****")
        if cfg.device == 'cuda':
            print(f"***** torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)} *****")

        self.work_dir = os.path.abspath(HydraConfig.get().run.dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        cfg.sampler_b = int(cfg.sampler_b)
        cfg.sampler_t = int(cfg.sampler_t)

        set_seed_everywhere(cfg.seed)  # NOTE: this part doesn't set seed for env
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(cfg.device)

        assert self.cfg.finetune_cfg.save_label_freq in [None, 1, -1]
        if self.cfg.finetune_cfg.save_label_freq is not None:
            assert self.cfg.finetune_cfg.finetune_model.label_capacity == None  # if label_capacity can not maintain all labels, label_buffer_statis will be incorrect
            self.label_buffer_statis = {
                                            'itr': [],
                                            'total_steps': [],
                                            'len_label': []
                                        }

        self.rank = 0  # from runner.startup()
        self.world_size = 1  # from runner.startup()
        self.global_B = cfg.sampler_b
        self.env_ranks = list(range(0, self.cfg.sampler_b))

        # From SerialSampler.__init__()
        self.batch_spec = BatchSpec(self.cfg.sampler_t, self.cfg.sampler_b)
        self.max_decorrelation_steps = 0
        self.mid_batch_reset = True  # immediately resets any environment which finishes an episode.
        # NOTE: self.CollectorCls = CpuResetCollector

        # From runner.__init__()
        self.max_steps = self.cfg.num_env_steps
        self.total_steps = 0
        self.eval_freq = self.cfg.eval_freq
        self.agent_save_freq = self.cfg.agent_save_freq

        if cfg.env.env_name.lower() in atari_env.ATARI_ENV:
            self.is_atari = True
            self.is_highway = False
            self.sampling_envs = [atari_env.make_env(cfg) for _ in range(cfg.sampler_b)]
            # env.seed(seed) is removed in gymnasium from v0.26 in favour of env.reset(seed)
            # see https://gymnasium.farama.org/content/migration-guide/#v21-to-v26-migration-guide for more information
            if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
                for t_senv in self.sampling_envs:
                    t_senv.unwrapped.save_human_img = True
            for id_env in range(cfg.sampler_b):
                self.sampling_envs[id_env].reset(seed=cfg.seed+id_env)
            
            self.eval_envs = [atari_env.make_env(cfg, eval=True) for _ in range(cfg.num_eval_episodes)]
            for id_env in range(cfg.num_eval_episodes):
                self.eval_envs[id_env].reset(seed=cfg.seed+id_env+123321778)

            self.total_env_lives = self.eval_envs[0].ale.lives()
            self.action_names = self.eval_envs[0].unwrapped.get_action_meanings()
            print(f'****** original total lives: {self.total_env_lives} ******')
            self.TrajInfoCls = GymAtariTrajInfo
            self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                            #  log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name,
                             reward_type=f'{cfg.finetune_cfg.type_name}_RLft',
                             traj_based=False,
                             env_name='atari')
        elif cfg.env.env_name in highway_env.HIGHWAY_ENV_NAME:
            self.is_atari = False
            self.is_highway = True
            self.sampling_envs = [highway_env.make_highway_env(cfg) for _ in range(cfg.sampler_b)]
            if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
                assert cfg.env.render_mode == 'rgb_array'
                for t_senv in self.sampling_envs:
                    t_senv.save_human_img = True
            for id_env in range(cfg.sampler_b):
                self.sampling_envs[id_env].reset(seed=cfg.seed+id_env)
            
            eval_env_cfg = copy.deepcopy(cfg)
            if cfg.save_eval_video.flag:
                eval_env_cfg.env.render_mode = 'rgb_array'
            self.eval_envs = [highway_env.make_highway_env(eval_env_cfg, eval=True) for _ in range(cfg.num_eval_episodes)]
            for id_env in range(cfg.num_eval_episodes):
                self.eval_envs[id_env].reset(seed=cfg.seed+id_env+123321778)
            self.action_names = self.eval_envs[0].action_type.actions
            self.TrajInfoCls = GymHighwayTrajInfo
            self.total_env_lives = 1
            self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                            #  log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name,
                             reward_type=f'{cfg.finetune_cfg.type_name}_RLft',
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
        
        # NOTE: we do not write SerialSampler explicitly, in order to make the code more consistent with PEBBLE
        cfg.agent.agent.model.action_dim = int(self.sampling_envs[0].action_space.n)
        cfg.agent.agent.model.obs_shape = self.sampling_envs[0].observation_space.shape  # (frame_stack, H, W)
        
        cfg.agent.algo.obs_shape = self.sampling_envs[0].observation_space.shape  # (frame_stack, H, W)
        if cfg.agent_model_cfg.encoder_cfg.is_mlp == False:
            cfg.agent_model_cfg.encoder_cfg.in_channels = self.sampling_envs[0].observation_space.shape[0]
        
        cfg.finetune_cfg.finetune_model.action_dim = int(self.sampling_envs[0].action_space.n)
        cfg.finetune_cfg.finetune_model.obs_shape = self.sampling_envs[0].observation_space.shape  # (frame_stack, H, W)

        if cfg.finetune_cfg.finetune_model.RL_loss.flag:
            cfg.agent.agent.separate_tgt = cfg.finetune_cfg.finetune_model.RL_loss.separate_tgt
            if cfg.finetune_cfg.finetune_model.RL_loss.separate_tgt:
                assert cfg.finetune_cfg.use_ft
        self.agent = hydra.utils.instantiate(cfg.agent.agent)
        # self.algo = hydra.utils.instantiate(cfg.agent.algo)
        self.agent.initialize(
                              # env_spaces=self.envs[0].spaces,
                              action_dim=self.sampling_envs[0].action_space.n,
                              # share_memory=False,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
        
        samples_pyt, samples_np, examples = build_samples_buffer(
            agent=self.agent,
            env=self.sampling_envs[0],
            batch_spec=self.batch_spec,
            agent_shared=False,
            env_shared=False,
            reward_type=f'{cfg.finetune_cfg.type_name}_RLft',
            )
        # NOTE: samples_pyt and samples_np share the same buffer
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        
        self.itr_batch_size = self.batch_spec.size
        # n_itr = int(self.max_steps // self.itr_batch_size)
        # self.n_itr = n_itr
        # print(f"------ Running {n_itr} iterations of fine-tuning. ------")
        
        if cfg.agent.agent.model.noisy:
            # TODO: I'm not sure if we should adding noise when using noisy-net for fine-tuning
            assert (self.cfg.finetune_cfg.noise_override is None) or\
                  (self.cfg.finetune_cfg.noise_override == False)
            # Note: if noise_override is None: finetuning = use noisy net, eval = no noisy net
            #                         if False:  both finetuning and eval will not use noisy net
            self.agent.model.head.set_noise_override(self.cfg.finetune_cfg.noise_override)
            self.agent.target_model.head.set_noise_override(self.cfg.finetune_cfg.noise_override)
        
        self.agent.to_device(self.device)
        self.agent.give_V_min_max(-self.cfg.agent.algo.V_max,
                                  self.cfg.agent.algo.V_max)
        # self.algo.initialize(
        #     agent=self.agent,
        #     n_itr=self.n_itr,
        #     batch_spec=self.batch_spec,
        #     mid_batch_reset=self.mid_batch_reset,
        #     examples=None,  # set None To avoid initializing algo's replay_buffer
        #     rank=self.rank,
        #     remove_frame_axis=env_remove_frame_axis,
        # )

        if cfg.finetune_cfg.type_name == 'CF':
            assert (cfg.finetune_cfg.oracle.exp_path is not None) and \
                    (cfg.finetune_cfg.oracle.ckpt_id is not None)
            # in fact, the constraints to buffer size is not so constrainted: just make sure that: when relabel return, the replay buffer is not full, otherwise relabel will be inaccurate
            # assert cfg.num_env_steps + cfg.agent.algo.n_step_return < cfg.agent.algo.replay_size  # Otherwise will have issue when relabel return
            with open(os.path.join(cfg.finetune_cfg.oracle.exp_path,
                                   'exp_logs/metadata.pkl'), 'rb') as f:
                oracle_hydra_cfg = pkl.load(f)
            print(f"oracle.env: {oracle_hydra_cfg['cfg']['env']}")
            print(f"current env: {cfg.env}")
            if self.is_atari:
                for k, v in oracle_hydra_cfg['cfg']['env'].items():
                    if k in ['clip_reward','terminal_on_life_loss','log_reward']:
                        print(f"oracle.env.clip_reward: {oracle_hydra_cfg['cfg']['env'][k]}; current clip_reward: {cfg.env[k]}")
                    else:
                        assert oracle_hydra_cfg['cfg']['env'][k] == cfg.env[k]
                for k, v in cfg.env.items():
                    if k not in oracle_hydra_cfg['cfg']['env'].keys():
                        assert k == 'log_reward'
            elif self.is_highway:
                for k, v in oracle_hydra_cfg['cfg']['env'].items():
                    if k == 'reward':
                        print(f"oracle.env.reward == cfg.env? : {oracle_hydra_cfg['cfg']['env'].reward==cfg.env.reward}")
                    elif k == 'act' and oracle_hydra_cfg['cfg']['env'].act.type=='DMeta':
                        if ('target_speeds' not in oracle_hydra_cfg['cfg']['env'].act.keys()):
                            assert cfg.env.act.target_speeds.st==20 and cfg.env.act.target_speeds.ed==30 and cfg.env.act.target_speeds.cnt==3
                        elif (oracle_hydra_cfg['cfg']['env']['act']['target_speeds']=='20-30-3'):
                            assert cfg.env.act.target_speeds.st==20 and cfg.env.act.target_speeds.ed==30 and cfg.env.act.target_speeds.cnt==3
                        else:
                            if oracle_hydra_cfg['cfg']['env'][k] != cfg.env[k]:
                                print(f"oracle's env config need to be consistent with your config: {oracle_hydra_cfg['cfg']['env'][k]} VS {cfg.env[k]} for {k}")
                                raise NotImplementedError
                    elif k in ['render_mode', 'show_traj', 'vehicles_density', 'vehicles_count', 'duration']:
                        if oracle_hydra_cfg['cfg']['env'][k] != cfg.env[k]:
                            print(f"oracle's env config is different with your config: {oracle_hydra_cfg['cfg']['env'][k]} VS {cfg.env[k]} for {k}")
                        continue
                    else:
                        if oracle_hydra_cfg['cfg']['env'][k] != cfg.env[k]:
                            print(f"oracle's env config need to be consistent with your config: {oracle_hydra_cfg['cfg']['env'][k]} VS {cfg.env[k]} for {k}")
                            raise NotImplementedError
                for k, v in cfg.env.items():
                    if k not in oracle_hydra_cfg['cfg']['env'].keys():
                        print(f"new key for current env config: key: {k}, value: {v}")
            else:
                raise NotImplementedError
            self.oracle_agent = hydra.utils.instantiate(oracle_hydra_cfg['cfg']['agent']['agent'])
            
            self.oracle_agent.initialize(
                              # env_spaces=self.envs[0].spaces,
                              action_dim=self.sampling_envs[0].action_space.n,
                            #   share_memory=False,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
            self.oracle_agent.load(model_dir=os.path.join(
                                        cfg.finetune_cfg.oracle.exp_path, 'exp_logs'),
                                    step=cfg.finetune_cfg.oracle.ckpt_id,
                                    device=self.device)
            # self.step = -1  # to distinguish from normal evaluation, use itr=-1 ans step=-1
            self.logger.log('eval/model_save_duration', -1, -1)  # stupid hack, to initialize a header in csv
            self.total_steps = -1  # to save correct log when evaluating oracle
            if self.cfg.finetune_cfg.finetune_model.cf_random > 0.:
                expert_eval_eps = self.cfg.finetune_cfg.finetune_model.cf_random
            else:
                expert_eval_eps = cfg.finetune_cfg.oracle.eps
            self.evaluate_agent(itr=-1, agent=self.oracle_agent,\
                                eval_eps=expert_eval_eps)
            self.total_steps = 0
            # self.oracle_agent.reset()
            self.oracle_agent.eval_mode(itr=1, eps=self.cfg.finetune_cfg.oracle.eps,
                                        verbose=True)  # NOTE: itr=1 s.t. the oracle also use eps_eval=0.001
        else:
            raise NotImplementedError
            self.oracle_agent = None

        self.finetune_type = cfg.finetune_cfg.type_name
        if cfg.finetune_cfg.type_name == 'CF':
            self.total_label = 0
            self.total_segment = 0
            self.total_itr = int(self.max_steps // self.batch_spec.T)  # NOTE: if #feedback_per_itr is not a constant w.r.t. itr, the finetuning procedure will stop early than total_itr
            print(f'[FINETUNE]: total_itr={self.total_itr}')
            cfg.finetune_cfg.finetune_model["total_itr"] = self.total_itr
            if cfg.finetune_cfg.log_q:
                cfg.finetune_cfg.finetune_model["log_q_path"] = os.path.join(
                                        self.work_dir, 'log_q')
            assert cfg.agent.algo.discount == 0.99  # currently I think the agent's discount should == oracle_agent's discount
            assert cfg.finetune_cfg.finetune_model.max_size is None
            if cfg.finetune_cfg.keep_all_data:
                cfg.finetune_cfg.finetune_model.max_size = cfg.num_env_steps + 200
            else:
                cfg.finetune_cfg.finetune_model.max_size = \
                    cfg.finetune_cfg.steps_per_itr * cfg.finetune_cfg.finetune_model.query_recent_itr + 200
                if cfg.finetune_cfg.finetune_model.RL_loss.flag:
                    cfg.finetune_cfg.finetune_model.max_size = max(
                            cfg.finetune_cfg.finetune_model.max_size,
                            cfg.finetune_cfg.steps_per_itr * cfg.finetune_cfg.finetune_model.RL_loss.RL_recent_itr
                        ) + 200
                    if (cfg.finetune_cfg.finetune_model.RL_loss.tgt_label.sl_weight > 0) and \
                        (not (cfg.finetune_cfg.finetune_model.RL_loss.tgt_label.same_data_RL or\
                              cfg.finetune_cfg.finetune_model.RL_loss.tgt_label.same_data_RL_target)):
                        cfg.finetune_cfg.finetune_model.max_size = max(
                            cfg.finetune_cfg.finetune_model.max_size,
                            cfg.finetune_cfg.steps_per_itr * \
                                cfg.finetune_cfg.finetune_model.RL_loss.tgt_label.data_recent_itr
                        ) + 200
                    if (cfg.finetune_cfg.finetune_model.RL_loss.RND_label.sl_weight > 0) and \
                        (not cfg.finetune_cfg.finetune_model.RL_loss.RND_label.same_data_RL):
                        cfg.finetune_cfg.finetune_model.max_size = max(
                            cfg.finetune_cfg.finetune_model.max_size,
                            cfg.finetune_cfg.steps_per_itr * \
                                cfg.finetune_cfg.finetune_model.RL_loss.RND_label.data_recent_itr
                        ) + 200
            # assert cfg.finetune_cfg.finetune_model.max_size >= \
            #         cfg.finetune_cfg.steps_per_itr * cfg.finetune_cfg.finetune_model.query_recent_itr + 200
            # if cfg.finetune_cfg.finetune_model.RL_loss.flag:
            #     cfg.finetune_cfg.finetune_model.max_size = max(cfg.finetune_cfg.finetune_model.max_size,
            #         cfg.finetune_cfg.steps_per_itr * cfg.finetune_cfg.finetune_model.RL_loss.data_recent_itr + 200)
            
            if not cfg.finetune_cfg.use_ft:
                assert cfg.finetune_cfg.finetune_model.RL_loss.flag
            
            if cfg.finetune_cfg.finetune_model.RL_loss.stop_acc is None:
                assert cfg.finetune_cfg.finetune_model.RL_loss.min_train_epoch == None
                # in this case, we use a max_train_epoch. To avoid confusion, assert min_train_epoch == None
            
            if cfg.finetune_cfg.finetune_model.label_capacity is None:
                if (cfg.finetune_cfg.finetune_model.ignore_small_qdiff.flag == False)\
                    and (cfg.finetune_cfg.finetune_model.ignore_small_qdiff.add_whole_seg == True):
                    cfg.finetune_cfg.finetune_model.label_capacity = \
                        self.total_itr * cfg.finetune_cfg.finetune_model.query_batch * cfg.finetune_cfg.finetune_model.size_segment\
                        + 200
                else:
                    cfg.finetune_cfg.finetune_model.label_capacity = \
                        self.total_itr * cfg.finetune_cfg.finetune_model.query_batch * cfg.finetune_cfg.finetune_model.cf_per_seg\
                        + 200
            
            if self.is_highway and self.cfg.env.obs.remove_frame_axis == True:
                env_remove_frame_axis = True
            else:
                env_remove_frame_axis = False

            cfg.finetune_cfg.finetune_model.log_path = self.work_dir
            self.finetune_model = hydra.utils.instantiate(cfg.finetune_cfg.finetune_model)
            self.finetune_model.initialize_buffer(  # build label buffer for reward model
                agent=self.agent,  # this agent will be saved by finetune_model.agent
                oracle=self.oracle_agent,
                env=self.sampling_envs[0],
                check_label_path=os.path.join(self.work_dir, 'CF_labels') \
                                 if self.cfg.finetune_cfg.check_label else None,
                segment_log_path=os.path.join(self.work_dir, 'segment_log'),
                action_names=self.action_names,
                remove_frame_axis=env_remove_frame_axis,
            )
            self.finetune_model.config_agent(agent=self.agent)
        else:
            raise NotImplementedError
        
        self.iterative_misspelling_check(self.cfg, token='Fasle')  # have mis-spell this config many times... 

        if self.cfg.finetune_cfg.eps_greedy.flag:
            self.sample_eps_ls = {}
        
        os.makedirs(self.work_dir, exist_ok=True)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))  # save cfg file

    def iterative_misspelling_check(self, dic, token):
        for key, value in dic.items():
            # print(f'key: {key}')
            if type(value) == DictConfig:
                self.iterative_misspelling_check(value, token)
            else:
                if value == token:
                    raise SyntaxError(f'Should use False instead of Fasle for {key} in cfg!!!')
    
    @torch.no_grad()
    def evaluate_agent_atari(self, itr, agent=None, eval_eps=None, special_logger=None):
        eval_begin = time.time()
        eval_completed_traj_infos, eps_this_eval = self.collect_evaluation_atari(itr, agent,
                                                        save_info=self.cfg.save_eval_img,
                                                        eval_eps=eval_eps)
        eval_end = time.time()
        eval_duration = eval_end - eval_begin
        if len(eval_completed_traj_infos) == 0:
            print("!!!!!WARNING: had no complete trajectories in eval at iter {itr}.!!!!!")

        logger = self.logger if special_logger is None else special_logger
        logger.log('eval/duration', eval_duration, self.total_steps)
        if agent is None:
            self._cum_eval_time += eval_duration
        
        logger.log('eval/num_completed_traj', len(eval_completed_traj_infos), self.total_steps)

        true_episode_reward_list = [info["Return"] for info in eval_completed_traj_infos]
        episode_reward_hat_list = [info["ReturnRHat"] for info in eval_completed_traj_infos]
        true_raw_episode_reward_list = [info["RawReturn"] for info in eval_completed_traj_infos]
        episode_length_list = [info["Length"] for info in eval_completed_traj_infos]

        for lst_name, stat_name in itertools.product(
                        ATARI_TRAJ_METRICS.keys(), RAINBOW_TRAJ_STATICS.keys()):
            logger.log(f'eval/{lst_name}_{stat_name}',
                             eval(f'np.{stat_name}({lst_name}_list)'), self.total_steps)
        
        logger.log('eval/eps_this_eval', eps_this_eval, self.total_steps)
        logger.log('eval/iter', itr, self.total_steps)
        logger.dump(self.total_steps, ty='eval')

    @torch.no_grad()
    def collect_evaluation_atari(self, itr, agent=None, save_info=False, eval_eps=None):
        check_eval_info_path=os.path.join(self.work_dir, f'check_eval_{itr}') \
                             if save_info else None
        if check_eval_info_path:
            os.makedirs(check_eval_info_path, exist_ok=True)
            assert len(self.eval_envs) == 1  # To save images easier
            entropy_ls = []
            obs_ls = []
            act_ls = []
            value_ls = []
            act_prob_ls = []
        if self.cfg.save_eval_video.flag == True:
            assert len(self.action_names) == self.cfg.agent.agent.model.action_dim
            eval_video_path = os.path.join(self.work_dir, f'eval_video_{self.cfg.env.env_name}_{itr}')
            os.makedirs(eval_video_path, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            text_hight = 30
            video_out_ls = [cv2.VideoWriter(f'{eval_video_path}/episode_{episode}.mp4',
                                       fourcc, float(self.cfg.save_eval_video.hz),
                                       (self.cfg.save_eval_video.shape_w, self.cfg.save_eval_video.shape_h + text_hight))
                                for episode in range(len(self.eval_envs))]
            human_img_ls = []
            terminated_ls = []
            text_font_size = 0.3
            text_action_pos = (1, 10)
            text_reward_pos = (1, 20)
            traj_info = {
                "true_episode_reward": np.zeros((len(self.eval_envs),)),
                "true_raw_episode_reward": np.zeros((len(self.eval_envs),)),
                "episode_length": np.zeros((len(self.eval_envs),)),
            }
        # If agent is not None, this function is used to evaluate oracle_Agent for CF reward
        eval_agent = agent if agent else self.agent
        eval_traj_infos = [
            self.TrajInfoCls(total_lives=self.total_env_lives) for _ in range(len(self.eval_envs))]
        completed_traj_infos = []
        observations = []
        for id_env, env in enumerate(self.eval_envs):
            env.unwrapped.save_human_img = True
            reset_obs, reset_info = env.reset()
            observations.append(reset_obs[:])
            if self.cfg.save_eval_video.flag == True:
                human_img_ls.append(reset_info["human_img"][:])
                terminated_ls.append(False)

        observation = buffer_from_example(observations[0], len(self.eval_envs))
        for b, o in enumerate(observations):
            observation[b] = o
        obs_pyt = torchify_buffer((observation))  # shape: (#eval_envs, C*frame_stack, H, W)

        # eval_agent.reset()
        eps_this_eval = eval_agent.eval_mode(itr=(1 if agent else itr),
                                             eps=eval_eps,
                                             verbose=True)  # itr > 1 will use eval_eps (or eps is specified), =0 will be random agent
        # if (agent is None) and self.reward_model:
        #     self.reward_model.eval()

        live_envs = list(range(len(self.eval_envs)))
        # for t in range(int(self.cfg.env.max_frames//4)):
        eval_t = 0
        while len(live_envs):  # Since all env has TimeLimit wrapper, all env will ended in limited time
            act_pyt, agent_info = eval_agent.step(obs_pyt)  # act_pyt.shape: (#eval_envs,), torch.int64
            action = numpify_buffer(act_pyt)
            assert action.shape == (len(live_envs),)

            if check_eval_info_path:
                obs_ls.append(copy.deepcopy(observation[0]))
                act_ls.append(action[0])
                value_ls.append(agent_info.value[0])
                act_prob = F.softmax(agent_info.value[0], dim=-1).cpu()
                act_prob_ls.append(act_prob)
                entropy_ls.append((-act_prob * torch.log(act_prob))\
                                    .sum(axis=-1, keepdims=False).item())

            # if (agent is None) and self.reward_model:  # When evaluate the loaded oracle, we pass the oracle agent here
            #     r_hat_sa = self.reward_model.r_hat_sa(obs_pyt, act_pyt)  # r_hat_sa_pyt.shape: (B,)
            # else:
            #     r_hat_sa = [None for _ in range(len(live_envs))]

            b = 0
            while b < len(live_envs):  # don't want to do a for loop since live envs changes over time
                env_id = live_envs[b]
                next_o, r, terminated, truncated, env_info = self.eval_envs[env_id].step(action[b])
                if self.cfg.save_eval_video.flag == True:
                    if terminated_ls[b]:
                        text_here = "Terminated"
                        color_here = (0, 0, 255)
                    else:
                        text_here = f"a_t: {self.eval_envs[env_id].action_names[action[b]]}"
                        color_here = (0, 0, 0)
                    save_img = cv2.putText(
                                        np.concatenate([
                                            (np.ones((text_hight, self.cfg.save_eval_video.shape_w, 3))*255).astype(np.uint8), # white color is (255, 255, 255)
                                            human_img_ls[b][:,:,::-1].copy(), # RGB to BGR, to be compatible with cv2
                                        ]), # RGB to BGR, to be compatible with cv2
                                        # f"a_t: {self.eval_envs[env_id].action_names[action[b]]}" if terminated_ls[b]==True else "Terminated",
                                        text_here,
                                        text_action_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                        color_here, 1)  # position, font, fontsize, fontcolor, fontweight
                    if r > 0:
                        reward_text_color = (0, 255, 0) # green
                    elif r < 0:
                        reward_text_color = (0, 0, 255) # red
                    else:
                        reward_text_color = (0, 0, 0) # black
                    save_img = cv2.putText(save_img.copy(),
                                        f"[R(s_t,a_t)]Sign: {int(r):d}, Raw: {int(env_info['raw_reward']):d}",  # R_t+1
                                        text_reward_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                        reward_text_color, 1)  # position, font, fontsize, fontcolor, fontweight
                    
                    video_out_ls[b].write(save_img.copy())

                eval_traj_infos[env_id].step(
                                reward=r,
                                r_hat=None,
                                raw_reward=env_info["raw_reward"],
                                terminated=terminated,
                                truncated=truncated,
                                need_reset=self.eval_envs[env_id].need_reset,
                                lives=self.eval_envs[env_id].ale.lives())

                if truncated or self.eval_envs[env_id].need_reset:
                    completed_traj_infos.append(eval_traj_infos[env_id].terminate())

                    observation = delete_ind_from_array(observation, b)
                    action = delete_ind_from_array(action, b)

                    obs_pyt = torchify_buffer((observation))
                    act_pyt = torchify_buffer((action))

                    if self.cfg.save_eval_video.flag == True:
                        text_here = "Truncated" if truncated else "Game Over"
                        save_img = cv2.putText(
                                        np.concatenate([
                                            (np.ones((text_hight, self.cfg.save_eval_video.shape_w, 3))*255).astype(np.uint8), # white color is (255, 255, 255)
                                            env_info["human_img"][:,:,::-1].copy(), # RGB to BGR, to be compatible with cv2
                                        ]),
                                        text_here,
                                        text_action_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                        (0, 0, 255), 1)  # position, font, fontsize, fontcolor, fontweight
                        if r > 0:
                            reward_text_color = (0, 255, 0) # green
                        elif r < 0:
                            reward_text_color = (0, 0, 255) # red
                        else:
                            reward_text_color = (0, 0, 0) # black
                        save_img = cv2.putText(save_img.copy(),
                                            f"[R(s_t,a_t)]Sign: {int(r):d}, Raw: {int(env_info['raw_reward']):d}",  # R_t+1
                                            text_reward_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                            reward_text_color, 1)  # position, font, fontsize, fontcolor, fontweight
                        video_out_ls[b].write(save_img.copy())
                        video_out_ls[b].release()
                        traj_info["true_episode_reward"][env_id] = completed_traj_infos[-1].Return  # only have raw reward for evaluation
                        traj_info["true_raw_episode_reward"][env_id] = completed_traj_infos[-1].RawReturn  # only have raw reward for evaluation
                        traj_info["episode_length"][env_id] = completed_traj_infos[-1].Length

                        del video_out_ls[b]
                        del human_img_ls[b]
                        del terminated_ls[b]
                        self.eval_envs[env_id].unwrapped.save_human_img = False
                    
                    del live_envs[b]
                    b -= 1  # live_envs[b] is now the next env, so go back one.
                else:
                    observation[b] = next_o[:]
                    if self.cfg.save_eval_video.flag == True:
                        human_img_ls[b] = env_info["human_img"][:]
                        terminated_ls[b] = terminated

                b += 1

                if len(completed_traj_infos) >= self.cfg.num_eval_episodes:
                    print("Evaluation reached max num trajectories "
                               f"({self.cfg.num_eval_episodes}).")
                    
                    if check_eval_info_path:
                        t_action_dim = self.cfg.agent.agent.model.action_dim
                        t_max_entropy = -t_action_dim * ((1.0/t_action_dim) * np.log((1.0/t_action_dim)))
                        rank = np.array(entropy_ls).reshape(-1).argsort().argsort()
                        rank_normalize = rank / len(obs_ls)
                        widths = [5] * (self.cfg.env.frame_stack + 2)
                        heights = [5,]
                        for tt in range(len(obs_ls)):
                            fig = plt.figure(constrained_layout=True,
                                            figsize=(len(widths)*5, len(heights)*5))
                            spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights),)
                            for img_id in range(self.cfg.env.frame_stack):
                                ax_img = fig.add_subplot(spec[0, img_id])
                                ax_plot_img(ori_img=obs_ls[tt][img_id],
                                            ax=ax_img,
                                            title=f'frame_stack: {tt}_{img_id}',
                                            vmin=0, vmax=255)
                            
                            ax_prob_bar = fig.add_subplot(spec[0, -2])
                            act_prob = act_prob_ls[tt]
                            bars = ax_plot_bar(ax=ax_prob_bar,
                                            xlabel=self.eval_envs[0].unwrapped.get_action_meanings(),
                                            height=act_prob)
                            for id_bar, bar in enumerate(bars):
                                yval = bar.get_height() 
                                ax_prob_bar.text(x=bar.get_x(), y=yval+.105,
                                        s=f'Q: {value_ls[tt][id_bar]:.3f}', rotation=45)
                            ax_prob_bar.axvline(x=bars[act_ls[tt]].get_x(), color='r')
                            ax_prob_bar.text(x=bars[act_ls[tt]].get_x()+.2,
                                                y=0.8,
                                                s=self.eval_envs[0].unwrapped.get_action_meanings()[act_ls[tt]],
                                                color='r')
                            
                            ax_entropy_color = fig.add_subplot(spec[0, -1])
                            ax_plot_heatmap(arr=np.array([rank_normalize[tt]]).reshape(1, 1),
                                            ax=ax_entropy_color,
                                            vmin=0., vmax=1.,
                                            s=f'entropy: {entropy_ls[tt]:.4f}/{t_max_entropy:.4f},\nrank: {rank[tt]}/{len(obs_ls)}',
                                            cmap=mpl.colormaps['YlGnBu'],
                                            text_color='r'
                                            )
                                            # highlight_row=target_oracle_index[id_seg].reshape(-1),
                                            # title='entropy (darkest <-> smallest)')  # normalized on its own min_max
                            fig.tight_layout()
                            fig.savefig(fname=os.path.join(check_eval_info_path,
                                                            f'{tt}.png'),
                                        bbox_inches='tight', pad_inches=0)

                    if self.cfg.save_eval_video.flag == True:
                        np.save(f"{eval_video_path}/traj_info.npy", traj_info)
                        cv2.destroyAllWindows()
                    return completed_traj_infos, eps_this_eval
            eval_t += 1

        if eval_t >= self.cfg.max_frames:
            print(f"!!!!!WARNING:Evaluation reached max num time steps {self.cfg.max_frames},",
                  f" but still have {len(live_envs)} env.!!!!!")

        return completed_traj_infos, eps_this_eval

    @torch.no_grad()
    def evaluate_agent_highway(self, itr, agent=None, eval_eps=None, special_logger=None):
        eval_begin = time.time()
        eval_completed_traj_infos, eps_this_eval = self.collect_evaluation_highway(itr, agent,
                                                        save_info=self.cfg.save_eval_img,
                                                        eval_eps=eval_eps)
        eval_end = time.time()
        eval_duration = eval_end - eval_begin
        if len(eval_completed_traj_infos) == 0:
            print("!!!!!WARNING: had no complete trajectories in eval at iter {itr}.!!!!!")
    
        logger = self.logger if special_logger is None else special_logger
        logger.log('eval/duration', eval_duration, self.total_steps)
        if agent is None:
            self._cum_eval_time += eval_duration
        
        # steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        # logger.record_tabular('StepsInEval', steps_in_eval)
        logger.log('eval/num_completed_traj', len(eval_completed_traj_infos), self.total_steps)
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
            logger.log(f'eval/{lst_name}_{stat_name}',
                             eval(f'np.{stat_name}({lst_name}_list)'), self.total_steps)
        
        logger.log('eval/eps_this_eval', eps_this_eval, self.total_steps)
        logger.log('eval/iter', itr, self.total_steps)
        logger.dump(self.total_steps, ty='eval')
    
    @torch.no_grad()
    def collect_evaluation_highway(self, itr, agent=None, save_info=False, eval_eps=None):
        check_eval_info_path=os.path.join(self.work_dir, f'check_eval_{itr}') \
                             if save_info else None
        if check_eval_info_path:
            os.makedirs(check_eval_info_path, exist_ok=True)
            assert len(self.eval_envs) == 1  # To save images easier
            entropy_ls = []
            obs_ls = []
            act_ls = []
            value_ls = []
            act_prob_ls = []
        if self.cfg.save_eval_video.flag == True:
            eval_video_path = os.path.join(self.work_dir, f'eval_video_{self.cfg.env.env_name}_{itr}')
            os.makedirs(eval_video_path, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out_ls = [cv2.VideoWriter(f'{eval_video_path}/episode_{episode}.mp4',
                                       fourcc, float(self.cfg.save_eval_video.hz),
                                       (self.cfg.save_eval_video.shape_w, self.cfg.save_eval_video.shape_h))
                                for episode in range(len(self.eval_envs))]
            human_img_ls = []
            text_action_pos = (5, 50)
            text_font_size = 0.5
            traj_info = {
                        "cnt_step": np.zeros((len(self.eval_envs),)),
                        "crashed": np.zeros((len(self.eval_envs),)),
                        "truncated": np.zeros((len(self.eval_envs),)),
                        "time": np.zeros((len(self.eval_envs),)),
                        "total_distance": np.zeros((len(self.eval_envs),)),
                        "cnt_lane_changed": np.zeros((len(self.eval_envs),)),
                        "total_right_lane_reward": np.zeros((len(self.eval_envs),)),
                        "total_high_speed_reward": np.zeros((len(self.eval_envs),)),
                        "total_low_speed_reward": np.zeros((len(self.eval_envs),)),
                    }
        
        # If agent is not None, this function is used to evaluate oracle_Agent for CF reward
        eval_agent = agent if agent else self.agent
        eval_traj_infos = [
            self.TrajInfoCls(total_lives=self.total_env_lives) for _ in range(len(self.eval_envs))]
        completed_traj_infos = []
        observations = []
        for id_env, t_env in enumerate(self.eval_envs):
            t_obs, t_info =  t_env.reset()
            observations.append(t_obs[:])
            if self.cfg.save_eval_video.flag == True:
                human_img_ls.append(t_info['human_img'][:])

        observation = buffer_from_example(observations[0], len(self.eval_envs))
        for b, o in enumerate(observations):
            observation[b] = o
        obs_pyt = torchify_buffer((observation))  # shape: (#eval_envs,D)

        # eval_agent.reset()
        eps_this_eval = eval_agent.eval_mode(itr=(1 if agent else itr),
                                             eps=eval_eps,
                                             verbose=True)  # itr > 1 will use eval_eps (or eps is specified), =0 will be random agent
        # if (agent is None) and self.reward_model:
        #     self.reward_model.eval()

        live_envs = list(range(len(self.eval_envs)))
        # for t in range(int(self.cfg.env.max_frames//4)):
        eval_t = 0
        while len(live_envs):  # Since all env has TimeLimit wrapper, all env will ended in limited time
            act_pyt, agent_info = eval_agent.step(obs_pyt)  # act_pyt.shape: (#eval_envs,), torch.int64
            action = numpify_buffer(act_pyt)
            assert action.shape == (len(live_envs),)

            if check_eval_info_path:
                obs_ls.append(copy.deepcopy(observation[0]))
                act_ls.append(action[0])
                value_ls.append(agent_info.value[0])
                act_prob = F.softmax(agent_info.value[0], dim=-1).cpu()
                act_prob_ls.append(act_prob)
                entropy_ls.append((-act_prob * torch.log(act_prob))\
                                    .sum(axis=-1, keepdims=False).item())

            # if (agent is None) and self.reward_model:  # When evaluate the loaded oracle, we pass the oracle agent here
            #     r_hat_sa = self.reward_model.r_hat_sa(obs_pyt, act_pyt)  # r_hat_sa_pyt.shape: (B,)
            # else:
            #     r_hat_sa = [None for _ in range(len(live_envs))]

            b = 0
            while b < len(live_envs):  # don't want to do a for loop since live envs changes over time
                env_id = live_envs[b]
                next_o, r, terminated, truncated, env_info = self.eval_envs[env_id].step(action[b])
                # print(f"action: {env_info['action_name']}, speed: {env_info['speed']}, crash: {env_info['crashed']}")
                # if env_info['crashed'] == 1:
                #     assert terminated
                if self.cfg.save_eval_video.flag == True:
                    text_here = f"action: {env_info['action_name']}, speed: {env_info['speed']:.3f}"
                    color_here = (0, 0, 0)
                    save_img = cv2.putText(human_img_ls[b][:,:,::-1].copy(), # RGB to BGR, to be compatible with cv2
                                            text_here,
                                            text_action_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                            color_here, 2)  # position, font, fontsize, fontcolor, fontweight
                    video_out_ls[b].write(save_img.copy())

                eval_traj_infos[env_id].step(
                                reward=r,
                                terminated=terminated,
                                truncated=truncated,
                                info=env_info)
                if truncated or terminated:
                    completed_traj_infos.append(eval_traj_infos[env_id].terminate())

                    observation = delete_ind_from_array(observation, b)
                    action = delete_ind_from_array(action, b)

                    obs_pyt = torchify_buffer((observation))
                    act_pyt = torchify_buffer((action))
                    # r_hat_sa_pyt = torchify_buffer((r_hat_sa))

                    del live_envs[b]

                    if self.cfg.save_eval_video.flag == True:
                        text_here = "Truncated"
                        save_img = cv2.putText(env_info['human_img'][:,:,::-1].copy(),  # RGB to BGR, to be compatible with cv2
                                            text_here,
                                            text_action_pos, cv2.FONT_HERSHEY_SIMPLEX, text_font_size,
                                            color_here, 2)  # position, font, fontsize, fontcolor, fontweight
                        video_out_ls[b].write(save_img.copy())
                        video_out_ls[b].release()
                        traj_info["cnt_step"][env_id] = completed_traj_infos[-1].Episodic_info["cnt_step"]
                        traj_info["crashed"][env_id] = completed_traj_infos[-1].Episodic_info["crashed"]
                        traj_info["truncated"][env_id] = completed_traj_infos[-1].Episodic_info["truncated"]
                        traj_info["time"][env_id] = completed_traj_infos[-1].Episodic_info["time"]
                        traj_info["total_distance"][env_id] = completed_traj_infos[-1].Episodic_info["total_distance"]
                        traj_info["cnt_lane_changed"][env_id] = completed_traj_infos[-1].Episodic_info["cnt_lane_changed"]
                        traj_info["total_right_lane_reward"][env_id] = completed_traj_infos[-1].Episodic_info["total_right_lane_reward"]
                        traj_info["total_high_speed_reward"][env_id] = completed_traj_infos[-1].Episodic_info["total_high_speed_reward"]
                        traj_info["total_low_speed_reward"][env_id] = completed_traj_infos[-1].Episodic_info["total_low_speed_reward"]

                        del video_out_ls[b]
                        del human_img_ls[b]

                    b -= 1  # live_envs[b] is now the next env, so go back one.
                else:
                    observation[b] = next_o[:]
                    if self.cfg.save_eval_video.flag == True:
                        human_img_ls[b] = env_info['human_img'][:]

                b += 1

                if len(completed_traj_infos) >= self.cfg.num_eval_episodes:
                    print("Evaluation reached max num trajectories "
                               f"({self.cfg.num_eval_episodes}).")
                    
                    if check_eval_info_path:
                        t_action_dim = self.cfg.agent.agent.model.action_dim
                        t_max_entropy = -t_action_dim * ((1.0/t_action_dim) * np.log((1.0/t_action_dim)))
                        rank = np.array(entropy_ls).reshape(-1).argsort().argsort()
                        rank_normalize = rank / len(obs_ls)
                        widths = [5] * (self.cfg.env.frame_stack + 2)
                        heights = [5,]
                        for tt in range(len(obs_ls)):
                            fig = plt.figure(constrained_layout=True,
                                            figsize=(len(widths)*5, len(heights)*5))
                            spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights),)
                                                    # width_ratios=widths,
                                                    # height_ratios=heights)
                            for img_id in range(self.cfg.env.frame_stack):
                                ax_img = fig.add_subplot(spec[0, img_id])
                                # ax_img = axes[id_in_seg, 0]
                                ax_plot_img(ori_img=obs_ls[tt][img_id],
                                            ax=ax_img,
                                            title=f'frame_stack: {tt}_{img_id}',
                                            vmin=0, vmax=255)
                            
                            ax_prob_bar = fig.add_subplot(spec[0, -2])
                            act_prob = act_prob_ls[tt]
                            bars = ax_plot_bar(ax=ax_prob_bar,
                                            xlabel=self.eval_envs[0].unwrapped.get_action_meanings(),
                                            height=act_prob)
                            for id_bar, bar in enumerate(bars):
                                yval = bar.get_height() 
                                ax_prob_bar.text(x=bar.get_x(), y=yval+.105,
                                        s=f'Q: {value_ls[tt][id_bar]:.3f}', rotation=45)
                            ax_prob_bar.axvline(x=bars[act_ls[tt]].get_x(), color='r')
                            ax_prob_bar.text(x=bars[act_ls[tt]].get_x()+.2,
                                                y=0.8,
                                                s=self.eval_envs[0].unwrapped.get_action_meanings()[act_ls[tt]],
                                                color='r')
                            
                            ax_entropy_color = fig.add_subplot(spec[0, -1])
                            ax_plot_heatmap(arr=np.array([rank_normalize[tt]]).reshape(1, 1),
                                            ax=ax_entropy_color,
                                            vmin=0., vmax=1.,
                                            s=f'entropy: {entropy_ls[tt]:.4f}/{t_max_entropy:.4f},\nrank: {rank[tt]}/{len(obs_ls)}',
                                            cmap=mpl.colormaps['YlGnBu'],
                                            text_color='r'
                                            )
                                            # highlight_row=target_oracle_index[id_seg].reshape(-1),
                                            # title='entropy (darkest <-> smallest)')  # normalized on its own min_max
                            fig.tight_layout()
                            fig.savefig(fname=os.path.join(check_eval_info_path,
                                                            f'{tt}.png'),
                                        bbox_inches='tight', pad_inches=0)

                    if self.cfg.save_eval_video.flag == True:
                        np.save(f"{eval_video_path}/traj_info.npy", traj_info)
                        cv2.destroyAllWindows()
                    return completed_traj_infos, eps_this_eval
            eval_t += 1

        if eval_t >= self.cfg.max_frames:
            print(f"!!!!!WARNING:Evaluation reached max num time steps {self.cfg.max_frames},",
                  f" but still have {len(live_envs)} env.!!!!!")
        if self.cfg.save_eval_video.flag == True:
            cv2.destroyAllWindows()
        return completed_traj_infos, eps_this_eval
    
    def finetune_agent(self,):
        if self.cfg.finetune_cfg.fb_schedule == 1:  # more steps, smaller frac
            frac = (self.max_steps - self.total_steps) / self.max_steps
            if frac == 0:
                frac = 0.01
        elif self.cfg.finetune_cfg.fb_schedule == 2:  # more steps, larger frac
            frac = self.max_steps / (self.max_steps - self.total_steps + 1)
        elif self.cfg.finetune_cfg.fb_schedule == 0:
            frac = 1
        else:
            raise NotImplementedError
        self.finetune_model.change_batch(frac)
        # if self.finetune_model.tgt_in_ft.flag==True and\  # remove the condition that finetune_model.tgt_in_ft.flag==True, since split_query can also be conducted when tgt_in_ft.flag==False 
        if self.finetune_model.tgt_in_ft.split_query==True:
            assert self.cfg.finetune_cfg.use_ft
            # change query_batch size
            if (self.itr >= self.finetune_model.tgt_in_ft.st_itr):
                if self.finetune_model.tgt_in_ft.flag==True:
                    self.finetune_model.ft_use_tgt = True
                self.finetune_model.set_batch(self.finetune_model.origin_query_batch//2)

        if (self.finetune_model.query_batch + self.total_segment > self.cfg.finetune_cfg.max_segment):
            self.finetune_model.set_batch(self.cfg.finetune_cfg.max_segment - self.total_segment)
        
        # get feedbacks (get_queries -> get_label -> put_queries)
        sample_st = time.time()
        _, collected_timestep = self.collect_batch(itr=self.itr)
        samples_to_data_buffer = self.finetune_model.samples_to_data_buffer(
                                                        self.samples_pyt)
        self.finetune_model.add_data(samples_to_data_buffer,
                                     collected_timestep=collected_timestep)
        if self.cfg.finetune_cfg.keep_all_data:
            assert not self.finetune_model._input_buffer_full
        
        if self.finetune_model.margine_decay.flag:
            current_margine = self.finetune_model.update_margine(itr=self.itr)
        
        cnt_labeled_queries = self.finetune_model.sampling(
                                itr=self.itr,
                                logger=self.logger,
                                log_total_steps=self.total_steps)
        if self.cfg.finetune_cfg.save_label_freq == 1:
            self.finetune_model.save_label_buffer(self.work_dir,
                                                 itr=self.itr+0.5,
                                                 total_steps=self.total_steps,
                                                 rewrite=self.cfg.finetune_cfg.save_label_rewrite)
        if self.cfg.finetune_cfg.save_label_freq is not None:
            self.label_buffer_statis['itr'].append(self.itr+0.5)
            self.label_buffer_statis['total_steps'].append(self.total_steps)
            self.label_buffer_statis['len_label'].append(self.finetune_model.len_label)

        sample_duration = time.time() - sample_st
        self.total_label += cnt_labeled_queries
        self.total_segment += self.finetune_model.query_batch
        
        finetune_st = time.time()
        if self.total_label > 0:
            model_update = self.cfg.finetune_cfg.model_update
            # TODO: repeat this loop for several times? then need to check how to log
            if self.cfg.finetune_cfg.use_ft:
                ft_tgt_oracle_act_acc_ls = []
                ft_agent_match_tgt_acc_ls = []
                ft_agent_match_oracle_acc_ls = []
                if self.finetune_model.reset_opt:
                    self.finetune_model.reset_opt_ft()
                for ft_epoch in range(model_update):
                    ft_train_human_acc, ft_train_losses, ft_train_grad_norms,\
                        ft_tgt_oracle_act_acc, ft_agent_match_tgt_acc, ft_agent_match_oracle_acc,\
                        ft_tgt_losses, ft_human_losses = \
                        self.finetune_model.finetune(itr=self.itr,
                                                     ft_epoch=ft_epoch)
                    
                    self.logger.log('interact_acc/epoch', ft_epoch, self.total_steps)
                    self.logger.log('interact_acc/human_acc', ft_train_human_acc, self.total_steps)
                    self.logger.log('interact_acc/loss', ft_train_losses, self.total_steps)
                    self.logger.log('interact_acc/grad_norm', ft_train_grad_norms, self.total_steps)
                    self.logger.log('interact_acc/tgt_oracle_act_acc', ft_tgt_oracle_act_acc, self.total_steps)
                    self.logger.log('interact_acc/agent_match_tgt_acc', ft_agent_match_tgt_acc, self.total_steps)
                    self.logger.log('interact_acc/agent_match_oracle_acc', ft_agent_match_oracle_acc, self.total_steps)
                    self.logger.log('interact_acc/tgt_losses', ft_tgt_losses, self.total_steps)
                    self.logger.log('interact_acc/human_losses', ft_human_losses, self.total_steps)
                    self.logger.log('interact_acc/ft_use_tgt', self.finetune_model.ft_use_tgt, self.total_steps)
                    self.logger.log('interact_acc/ft_w_human', self.finetune_model.ft_w_human_ls[-1], self.total_steps)
                    self.logger.log('interact_acc/ft_w_tgt', self.finetune_model.ft_w_tgt_ls[-1], self.total_steps)
                    self.logger.dump(self.total_steps, ty='interact_acc')

                    if ft_tgt_oracle_act_acc is not None:
                        ft_tgt_oracle_act_acc_ls.append(ft_tgt_oracle_act_acc)
                        ft_agent_match_tgt_acc_ls.append(ft_agent_match_tgt_acc)
                        ft_agent_match_oracle_acc_ls.append(ft_agent_match_oracle_acc)
                    
                    ft_target_acc = self.cfg.finetune_cfg.acc_target
                    if (self.finetune_model.ft_use_tgt) and \
                        (self.cfg.finetune_cfg.finetune_model.tgt_in_ft.acc_target is not None):
                        if (ft_train_human_acc > ft_target_acc) and\
                            (ft_agent_match_tgt_acc > self.cfg.finetune_cfg.finetune_model.tgt_in_ft.acc_target):
                            break
                    else:
                        if ft_train_human_acc > ft_target_acc:
                            break
                
                if (self.itr+1) % self.eval_freq == 0:
                    self.evaluate_agent(itr=self.itr+0.5,
                                    eval_eps=self.cfg.agent.agent.eps_eval)
            
            if self.finetune_model.RL_loss.flag:
                # firstly, consider if tgt_in_ft.split_query==True
                # if self.finetune_model.tgt_in_ft.flag==True and\
                #     self.finetune_model.ft_use_tgt==True and\
                if self.finetune_model.tgt_in_ft.split_query==True and\
                    (self.itr >= self.finetune_model.tgt_in_ft.st_itr):
                    # get feedbacks (get_queries -> get_label -> put_queries)
                    sample_st = time.time()
                    _, collected_timestep = self.collect_batch(itr=self.itr+0.5)
                    samples_to_data_buffer = self.finetune_model.samples_to_data_buffer(
                                                                    self.samples_pyt)
                    self.finetune_model.add_data(samples_to_data_buffer,
                                                collected_timestep=collected_timestep)
                    if self.cfg.finetune_cfg.keep_all_data:
                        assert not self.finetune_model._input_buffer_full
                    
                    if self.finetune_model.margine_decay.flag:
                        raise NotImplementedError
                        current_margine = self.finetune_model.update_margine(itr=self.itr)
                    assert self.finetune_model.query_recent_itr == 1  # for other case, need to consider how to maintain finetune_model.T_itr_ls
                    cnt_labeled_queries_half = self.finetune_model.sampling(
                                            itr=self.itr+0.5,
                                            logger=self.logger,
                                            log_total_steps=self.total_steps)
                    if self.cfg.finetune_cfg.save_label_freq == 1:
                        self.finetune_model.save_label_buffer(self.work_dir,
                                                            itr=self.itr+1,
                                                            total_steps=self.total_steps)
                    if self.cfg.finetune_cfg.save_label_freq is not None:
                        self.label_buffer_statis['itr'].append(self.itr+1)
                        self.label_buffer_statis['total_steps'].append(self.total_steps)
                        self.label_buffer_statis['len_label'].append(self.finetune_model.len_label)

                    sample_duration += time.time() - sample_st
                    cnt_labeled_queries += cnt_labeled_queries_half
                    self.total_label += cnt_labeled_queries_half
                    self.total_segment += self.finetune_model.query_batch
                    self.finetune_model.merge_T_itr_ls(merge_cnt=2)  # merge the latest 2 number
                
                if self.finetune_model.RL_loss.separate_tgt and \
                    (self.finetune_model.RL_loss.separate_update_tgt_interval is None):
                    self.finetune_model.agent.update_separate_target(tau=self.finetune_model.RL_loss.separate_target_tau)
                if (self.finetune_model.RL_loss.tgt_label.sl_weight > 0) and\
                    (self.finetune_model.RL_loss.tgt_label.RND_check.filter is not None):
                    tgt_RND_confident_ratio_epoch_ls = []
                    wrong_doubt_ratio_epoch_ls = []
                
                max_RL_epoch = self.cfg.finetune_cfg.finetune_model.RL_loss.max_train_epoch
                if self.finetune_model.ft_use_tgt:
                    max_RL_epoch = max_RL_epoch if \
                                    (self.cfg.finetune_cfg.finetune_model.tgt_in_ft.RL_epoch is None) else\
                                    self.cfg.finetune_cfg.finetune_model.tgt_in_ft.RL_epoch
                elif self.cfg.finetune_cfg.finetune_model.tgt_in_ft.flag == False:
                    if self.itr >= self.cfg.finetune_cfg.finetune_model.tgt_in_ft.st_itr:
                        max_RL_epoch = max_RL_epoch if \
                                    (self.cfg.finetune_cfg.finetune_model.tgt_in_ft.RL_epoch is None) else\
                                    self.cfg.finetune_cfg.finetune_model.tgt_in_ft.RL_epoch
                if self.finetune_model.reset_opt:
                    self.finetune_model.reset_opt_rl()
                for RL_epoch in range(max_RL_epoch):
                    rl_1_loss_avg, rl_n_loss_avg, loss_avg, grad_norms_avg,\
                        sl_human_loss_avg, sl_tgt_loss_avg, sl_RND_loss_avg,\
                        CF_agent_pred_human_acc, CF_agent_pred_tgt_acc, CF_agent_pred_RND_acc,\
                            tgt_RND_info = \
                        self.finetune_model.RL_finetune(upd_epoch=RL_epoch)
                    self.logger.log('rlloss_epoch/epoch', RL_epoch, self.total_steps)
                    if CF_agent_pred_human_acc is not None:
                        self.logger.log('rlloss_epoch/CF_agent_pred_human_acc', CF_agent_pred_human_acc, self.total_steps)
                    if CF_agent_pred_tgt_acc is not None:
                        self.logger.log('rlloss_epoch/CF_agent_pred_tgt_acc', CF_agent_pred_tgt_acc, self.total_steps)
                    if CF_agent_pred_RND_acc is not None:
                        self.logger.log('rlloss_epoch/CF_agent_pred_RND_acc', CF_agent_pred_RND_acc, self.total_steps)
                    if tgt_RND_info is not None:
                        tgt_RND_confident_ratio_epoch_ls.append(tgt_RND_info['tgt_RND_confident_ratio'])
                        wrong_doubt_ratio_epoch_ls.append(tgt_RND_info['wrong_doubt_ratio'])
                        self.logger.log('rlloss_epoch/tgt_RND_confident_ratio',
                                         tgt_RND_info['tgt_RND_confident_ratio'], self.total_steps)
                        self.logger.log('rlloss_epoch/wrong_doubt_ratio',
                                         tgt_RND_info['wrong_doubt_ratio'], self.total_steps)
                    else:
                        self.logger.log('rlloss_epoch/tgt_RND_confident_cnt', None, self.total_steps)
                        self.logger.log('rlloss_epoch/num_wrong_doubt', None, self.total_steps)
                    self.logger.log('rlloss_epoch/rl_1_loss_avg', rl_1_loss_avg, self.total_steps)
                    self.logger.log('rlloss_epoch/rl_n_loss_avg', rl_n_loss_avg, self.total_steps)
                    self.logger.log('rlloss_epoch/sl_human_loss_avg', sl_human_loss_avg, self.total_steps)
                    self.logger.log('rlloss_epoch/sl_tgt_loss_avg', sl_tgt_loss_avg, self.total_steps)
                    self.logger.log('rlloss_epoch/sl_RND_loss_avg', sl_RND_loss_avg, self.total_steps)
                    self.logger.log('rlloss_epoch/loss_avg', loss_avg, self.total_steps)
                    self.logger.log('rlloss_epoch/grad_norms_avg', grad_norms_avg, self.total_steps)
                    self.logger.dump(self.total_steps, ty='rlloss_epoch')

                    stop_acc = self.cfg.finetune_cfg.finetune_model.RL_loss.stop_acc
                    if (stop_acc is not None)\
                            and ((RL_epoch + 1) >= self.cfg.finetune_cfg.finetune_model.RL_loss.min_train_epoch):
                        reach_acc = True
                        if CF_agent_pred_human_acc is not None:
                            reach_acc &= (CF_agent_pred_human_acc >= stop_acc)
                        if CF_agent_pred_tgt_acc is not None:
                            reach_acc &= (CF_agent_pred_tgt_acc >= stop_acc)
                        if CF_agent_pred_RND_acc is not None:
                            reach_acc &= (CF_agent_pred_RND_acc >= stop_acc)
                        if reach_acc:
                            break

                self.logger.log('rlloss/epoch_cnt', RL_epoch, self.total_steps)
                if CF_agent_pred_human_acc is not None:
                    self.logger.log('rlloss/CF_agent_pred_human_acc', CF_agent_pred_human_acc, self.total_steps)
                if CF_agent_pred_tgt_acc is not None:
                    self.logger.log('rlloss/CF_agent_pred_tgt_acc', CF_agent_pred_tgt_acc, self.total_steps)
                if CF_agent_pred_RND_acc is not None:
                    self.logger.log('rlloss/CF_agent_pred_RND_acc', CF_agent_pred_RND_acc, self.total_steps)
                self.logger.log('rlloss/rl_1_loss_avg', rl_1_loss_avg, self.total_steps)
                self.logger.log('rlloss/rl_n_loss_avg', rl_n_loss_avg, self.total_steps)
                self.logger.log('rlloss/sl_human_loss_avg', sl_human_loss_avg, self.total_steps)
                self.logger.log('rlloss/sl_tgt_loss_avg', sl_tgt_loss_avg, self.total_steps)
                self.logger.log('rlloss/sl_RND_loss_avg', sl_RND_loss_avg, self.total_steps)
                self.logger.log('rlloss/loss_avg', loss_avg, self.total_steps)
                self.logger.log('rlloss/grad_norms_avg', grad_norms_avg, self.total_steps)
                if tgt_RND_info is not None:
                    self.logger.log('rlloss/tgt_RND_confident_ratio_avg',
                                    np.mean(tgt_RND_confident_ratio_epoch_ls), self.total_steps)
                    self.logger.log('rlloss/wrong_doubt_ratio_avg',
                                    np.mean(wrong_doubt_ratio_epoch_ls), self.total_steps)
                else:
                    self.logger.log('rlloss/tgt_RND_confident_ratio_avg', None, self.total_steps)
                    self.logger.log('rlloss/wrong_doubt_ratio_avg', None, self.total_steps)
                self.logger.dump(self.total_steps, ty='rlloss')
                
                self.logger.log('interact/unique_CF_labels', sum(self.finetune_model.have_label_flag_buffer).item(), self.total_steps)  # have_label_flag_buffer only for RL_loss.flag==True 
            
            if self.cfg.finetune_cfg.use_ft:  # to maintain a same 'step' flag for interact's log when tgt_in_ft.split_query==True, this part are logged after the RL-phase, where another collect_data may be called
                # print(f"[R_HAT] Reward function in step {self.total_steps} is updated!! ACC: " + str(ft_train_human_acc))
                self.logger.log('interact/total_update', ft_epoch+1, self.total_steps)
                self.logger.log('interact/final_human_acc', ft_train_human_acc, self.total_steps)
                self.logger.log('interact/final_loss', ft_train_losses, self.total_steps)
                self.logger.log('interact/q_diff_average', self.finetune_model.q_diff_average, self.total_steps)
                if len(ft_tgt_oracle_act_acc_ls) > 0:
                    self.logger.log('interact/tgt_oracle_act_acc_avg', 
                                    np.mean(ft_tgt_oracle_act_acc_ls), self.total_steps)
                    self.logger.log('interact/agent_match_tgt_acc_avg', 
                                    np.mean(ft_agent_match_tgt_acc_ls), self.total_steps)
                    self.logger.log('interact/agent_match_oracle_acc_avg', 
                                    np.mean(ft_agent_match_oracle_acc_ls), self.total_steps)
                else:
                    self.logger.log('interact/tgt_oracle_act_acc_avg', 0., self.total_steps)
                    self.logger.log('interact/agent_match_tgt_acc_avg', 0., self.total_steps)
                    self.logger.log('interact/agent_match_oracle_acc_avg', 0., self.total_steps)
                # self.logger.log('interact/tgt_test_acc', self.finetune_model.tgt_test_acc, self.total_steps)
                # self.logger.log('interact/final_tgt_act_acc', tgt_oracle_act_acc, self.total_steps)
                self.logger.log('interact/final_tgt_losses', ft_tgt_losses, self.total_steps)
                self.logger.log('interact/final_human_losses', ft_human_losses, self.total_steps)
                self.logger.log('interact/ft_use_tgt', self.finetune_model.ft_use_tgt, self.total_steps)
                self.logger.log('interact/ft_w_human_avg',
                                    np.mean(self.finetune_model.ft_w_human_ls), self.total_steps)
                self.logger.log('interact/ft_w_tgt_avg',
                                    np.mean(self.finetune_model.ft_w_tgt_ls), self.total_steps)
                self.logger.log('interact/ft_human_train_bs',
                                        self.finetune_model.ft_human_train_bs, self.total_steps)
                self.logger.log('interact/ft_tgt_train_bs',
                                    self.finetune_model.ft_tgt_train_bs, self.total_steps)
                if self.finetune_model.margine_decay.flag:
                    self.logger.log('interact/loss_margine', current_margine, self.total_steps)
        else:
            raise NotImplementedError  # In fact, in this case, RL-phase should be put outside this loop: we should allow pure RL phase after all human labels are provided
            self.logger.log('interact/total_update', 0, self.total_steps)
            # print(f"[R_HAT] No labeled feedback to train reward function in step {self.total_steps}")
        finetune_duration = time.time() - finetune_st
        return cnt_labeled_queries, sample_duration, finetune_duration
    
    def run(self):
        # From SerialSampler.initialize()
        self.traj_infos = [self.TrajInfoCls(total_lives=self.total_env_lives)\
                            for _ in range(len(self.sampling_envs))]
        
        obs_ls = list()
        if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
            obs_human_img_ls = list()
        
        for env in self.sampling_envs:
            reset_obs, reset_info = env.reset()
            obs_ls.append(reset_obs[:])
            if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
                obs_human_img_ls.append(reset_info["human_img"][:])
        
        observation = buffer_from_example(obs_ls[0], len(self.sampling_envs))
        for b, obs in enumerate(obs_ls):
            observation[b] = obs  # numpy array or namedarraytuple
        self.agent_inputs = observation

        if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
            obs_human_img = buffer_from_example(obs_human_img_ls[0], len(self.sampling_envs))
            for b, obs_hm in enumerate(obs_human_img_ls):
                obs_human_img[b] = obs_hm  # numpy array or namedarraytuple
            self.obs_hm_t = obs_human_img
        
        self._cum_eval_time = 0
        global_start_time = time.time()

        # evaluation
        self.itr = 0
        allow_update = True
        while allow_update:
            if self.itr > 0 and self.agent_save_freq > 0 and \
                    self.itr % self.agent_save_freq == 0:
                save_model_start = time.time()
                self.agent.save(self.work_dir, self.total_steps)
                save_model_end = time.time()
                self.logger.log('eval/model_save_duration', save_model_end-save_model_start, self.total_steps)
            if self.itr % self.eval_freq == 0:
                if self.itr == 0 and not self.cfg.eval_at_itr0:
                    pass
                else:
                    if self.itr == 0:
                        self.evaluate_agent(itr=self.itr,
                            eval_eps=self.cfg.agent.agent.eps_eval \
                                if self.cfg.agent.agent.ckpt_path is not None else 1.)
                    else:
                        self.evaluate_agent(itr=self.itr,
                            eval_eps=self.cfg.agent.agent.eps_eval)
                        if self.cfg.finetune_cfg.SL_end_cfg.flag:
                            self.SL_on_labels_end()
            label_this_itr, sample_duration, finetune_duration = self.finetune_agent()  # collect data & finetune_agent
            self.logger.log('interact/sample_duration', sample_duration, self.total_steps)
            self.logger.log('interact/finetune_duration', finetune_duration, self.total_steps)

            # one feedback may not be counted as one label. E.g., render two segments but the expert think it's useless to compare the two
            self.logger.log('interact/total_segment', self.total_segment, self.total_steps)
            self.logger.log('interact/segment_this_itr', self.finetune_model.query_batch, self.total_steps)
            self.logger.log('interact/total_label', self.total_label, self.total_steps)
            self.logger.log('interact/label_this_itr', label_this_itr, self.total_steps)
            self.logger.log('interact/len_inputs_T', self.finetune_model.len_inputs_T, self.total_steps)
            self.logger.log('interact/len_label', self.finetune_model.len_label, self.total_steps)
            self.logger.log('interact/iter', self.itr+1, self.total_steps)
            if self.cfg.finetune_cfg.eps_greedy.flag:
                self.logger.log('interact/sample_eps', self.sample_eps_ls[self.itr], self.total_steps)  # have_label_flag_buffer only for RL_loss.flag==True 
            # log for ft-phase
            self.logger.dump(self.total_steps, ty='interact')

            allow_update = (self.total_steps < self.max_steps) and\
                           (self.total_segment < self.cfg.finetune_cfg.max_segment)
            self.itr += 1
        if self.cfg.finetune_cfg.log_q:
            self.finetune_model.put_qlog(self.itr, None, None, None, None)

        # self.agent.train_mode(self.itr)
        # opt_info = self.algo.optimize_agent(self.itr)
        
        # self.step = (itr + 1) * self.batch_spec.size
        # self._cum_completed_trajs += len(completed_infos)
        # itr_train_end_time = time.time()
    
        global_end_time = time.time() 
        print(f'[Total Running Time]: {global_end_time - global_start_time}')

        try:
            self.agent.save(self.work_dir, self.total_steps)
        except:
            pass
        
        final_end_time = time.time()
        self.logger.log('eval/model_save_duration', final_end_time - global_end_time, self.total_steps)

        self.evaluate_agent(itr=self.itr,
                            eval_eps=self.cfg.agent.agent.eps_eval)
        if self.cfg.finetune_cfg.SL_end_cfg.flag:
            self.SL_on_labels_end()
        
        if self.cfg.finetune_cfg.log_q:
            self.finetune_model.plot_q(cnt_samples=min(1000, self.total_label//20),
                                       action_names=self.action_names)
        
        if self.cfg.finetune_cfg.save_label_freq == -1:
            self.finetune_model.save_label_buffer(self.work_dir,
                                                 itr=self.itr,
                                                 total_steps=self.total_steps)
        if self.cfg.finetune_cfg.save_label_freq is not None:
            with open(os.path.join(self.work_dir,
                      f'label_buffer_statis.pkl'),
                        'wb') as file:
                pkl.dump(self.label_buffer_statis, file)

    def SL_on_labels_end(self,):
        self._cum_eval_time = 0
        SL_cfg = self.cfg.finetune_cfg.SL_end_cfg
        SL_dir = os.path.join(self.work_dir, 'SL_logs')
        os.makedirs(SL_dir, exist_ok=True)
        SL_logger = Logger(SL_dir,
                            save_tb=False,
                            agent='SL',
                            reward_type='SL',
                            traj_based=False,
                            env_name='atari')
        
        SL_agent = hydra.utils.instantiate(self.cfg.agent.agent)
        SL_agent.initialize(action_dim=self.cfg.agent.agent.model.action_dim,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
        if self.cfg.agent.agent.model.noisy:
            # TODO: I'm not sure if we should adding noise when using noisy-net for fine-tuning
            assert (self.cfg.finetune_cfg.noise_override is None) or\
                  (self.cfg.finetune_cfg.noise_override == False)
            # Note: if noise_override is None: finetuning = use noisy net, eval = no noisy net
            #                         if False:  both finetuning and eval will not use noisy net
            SL_agent.model.head.set_noise_override(self.cfg.finetune_cfg.noise_override)
            SL_agent.target_model.head.set_noise_override(self.cfg.finetune_cfg.noise_override)
        SL_agent.to_device(self.device)
        SL_agent.give_V_min_max(-self.cfg.agent.algo.V_max,
                                  self.cfg.agent.algo.V_max)
        
        label_buffer = self.finetune_model.label_buffer[:self.finetune_model.len_label]
        SL_ft_model = hydra.utils.instantiate(self.cfg.finetune_cfg.finetune_model)
        SL_ft_model.config_agent(SL_agent)  # will config agent & optimizer for that agent's parameters here
        SL_ft_model.reset_label_buffer(label_buffer)

        if SL_cfg.eval_thres == True:
            SL_cfg.eval_acc_thres_ls = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9,
                                    0.925, 0.95, 0.975, 0.98, 0.99, 0.999]
        else:
            SL_cfg.eval_acc_thres_ls = [1.1]
        SL_eval_acc_thres_ls = np.array(SL_cfg.eval_acc_thres_ls)
        
        SL_ft_model.ft_use_tgt = False
        SL_logger.log('eval/acc_thres', 0, 0)
        SL_logger.log('eval/human_acc', 0, 0)
        SL_logger.log('eval/sl_epoch', 0, 0)
        self.sl_epoch = 0
        if self.cfg.finetune_cfg.SL_end_cfg.eval_SL0:
            self.evaluate_agent(itr=0,
                                # epoch=0,
                                agent=SL_agent,
                                eval_eps=self.cfg.agent.agent.eps_eval,
                                special_logger=SL_logger)
        
        for sl_epoch in range(1, SL_cfg.model_update):  # epoch count starting from 1 to avoid tgt_in_ft settings in finetune_model.finetune()
            self.sl_epoch = sl_epoch
            ft_train_human_acc, ft_train_losses, ft_train_grad_norms,\
                _, _, _,_, _ = SL_ft_model.finetune(itr=-1,
                                                    ft_epoch=sl_epoch)
            if self.cfg.finetune_cfg.SL_end_cfg.log_SL:
                SL_logger.log('SL_epoch/epoch', sl_epoch, sl_epoch)
                SL_logger.log('SL_epoch/human_acc', ft_train_human_acc, sl_epoch)
                SL_logger.log('SL_epoch/loss', ft_train_losses, sl_epoch)
                SL_logger.log('SL_epoch/grad_norm', ft_train_grad_norms, sl_epoch)
                SL_logger.dump(sl_epoch, ty='SL_epoch')

            evaled = False
            if sl_epoch == 1:
                st_acc_thres_idx = np.sum(SL_eval_acc_thres_ls < ft_train_human_acc)
            if (st_acc_thres_idx < len(SL_eval_acc_thres_ls)) and\
                ft_train_human_acc >= SL_eval_acc_thres_ls[st_acc_thres_idx]:
                SL_logger.log('eval/acc_thres', SL_eval_acc_thres_ls[st_acc_thres_idx], sl_epoch)
                SL_logger.log('eval/human_acc', ft_train_human_acc, sl_epoch)
                SL_logger.log('eval/sl_epoch', sl_epoch, sl_epoch)
                self.evaluate_agent(itr=1,
                                # epoch=sl_epoch,
                                agent=SL_agent,
                                eval_eps=self.cfg.agent.agent.eps_eval,
                                special_logger=SL_logger)
                st_acc_thres_idx += 1
                evaled = True
            
            if ft_train_human_acc >= SL_cfg.acc_target and\
                st_acc_thres_idx == len(SL_eval_acc_thres_ls):
                break
        
        if not evaled:
            SL_logger.log('eval/acc_thres', ft_train_human_acc, sl_epoch)
            SL_logger.log('eval/human_acc', ft_train_human_acc, sl_epoch)
            SL_logger.log('eval/sl_epoch', sl_epoch, sl_epoch)
            self.evaluate_agent(itr=1,
                            # epoch=sl_epoch,
                            agent=SL_agent,
                            eval_eps=self.cfg.agent.agent.eps_eval,
                            special_logger=SL_logger)

    def collect_batch(self, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        self.agent.eps_sample = 0.0
        if self.cfg.finetune_cfg.eps_greedy.flag:
            eps_greedy_cfg = self.cfg.finetune_cfg.eps_greedy
            if itr >= eps_greedy_cfg.eps_itr:
                self.sample_eps_ls[itr] = eps_greedy_cfg.ed_eps
            else:
                self.sample_eps_ls[itr] = min(eps_greedy_cfg.init_eps, max(eps_greedy_cfg.ed_eps,
                                          eps_greedy_cfg.ed_eps+((eps_greedy_cfg.init_eps - eps_greedy_cfg.ed_eps) / (1.0 * eps_greedy_cfg.eps_itr)) * (eps_greedy_cfg.eps_itr - itr)))
        if self.cfg.finetune_cfg.traj_collect.mode == 'eval':
            self.agent.eval_mode(itr=itr,  # itr is useless because eps for evaluation is irrelevant with itr when the eps argument is set
                                 eps=self.cfg.agent.agent.eps_eval if not self.cfg.finetune_cfg.eps_greedy.flag\
                                    else self.sample_eps_ls[itr],
                                 verbose=True)
            if self.agent.model.noisy:
                reset_noise_interval = self.batch_spec.T  # do not need to reset
        elif self.cfg.finetune_cfg.traj_collect.mode == 'sample':
            # TODO: do we need to reset_noise
            self.agent.sample_mode(itr=10000, verbose=True)  # NOTE: a large itr to avoid using epislon greedy
            if self.agent.model.noisy:
                reset_noise_interval = self.cfg.finetune_cfg.traj_collect.reset_noise_interval
        else:
            raise NotImplementedError
        
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env  # leading_dims: [T, B]
        
        oracle_act_buf = self.samples_np.oracle_act
        oracle_act_prob_buf = self.samples_np.oracle_act_prob
        oracle_q_buf = self.samples_np.oracle_q
        
        completed_infos = list()
        observation = self.agent_inputs  # observation.shape = (B, frame_stack, H, W)
        obs_pyt = torchify_buffer(self.agent_inputs)  # share self.agent_inputs's memory
        
        collected_timestep = self.batch_spec.T
        # if self.finetune_model.tgt_in_ft.flag==True and\
        #    self.finetune_model.ft_use_tgt==True and\
        if (self.finetune_model.tgt_in_ft.split_query==True) and\
            (self.itr >= self.finetune_model.tgt_in_ft.st_itr):  # NOTE: if you modify this condition, you need also check the later half part in finetune_agent(), which use a same condition as here
            collected_timestep = collected_timestep//2

        for t in range(collected_timestep):
            if self.agent.model.noisy and t % reset_noise_interval == 0:
                self.agent.model.head.reset_noise()

            env_buf.observation[t] = observation  # slice 
            if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
                env_buf.human_img[t] = self.obs_hm_t
            # NOTE: "env_buf.observation[t]" will not be modified after "observation" being updated
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt)  # NOTE: no_grad for agent.step
            # if self.reward_model:
                # r_hat_sa = self.reward_model.r_hat_sa(obs_pyt, act_pyt)  # r_hat_sa.shape: (B,)
                # if self.oracle_agent:
            oracle_act_pyt, oracle_act_info = self.oracle_agent.step(obs_pyt)
            # oracle_act_pyt.shape: (B,), oracle_act_info.p.shape: (B, action_dim)

            action = numpify_buffer(act_pyt)  # shape (B,)
            agent_buf.action[t] = action
            # if self.reward_model:
            #     r_hat_buf[t] = r_hat_sa  # NOTE: changing this buffer will change samples_np & samples_pyt at the same time
            #     if self.oracle_agent:
            oracle_act = numpify_buffer(oracle_act_pyt)
            oracle_act_buf[t] = oracle_act

            oracle_act_prob_pyt = F.softmax(
                oracle_act_info.value / self.cfg.finetune_cfg.finetune_model.softmax_tau,
                dim=-1)
            oracle_act_prob = numpify_buffer(oracle_act_prob_pyt)
            oracle_act_prob_buf[t] = oracle_act_prob

            oracle_q = numpify_buffer(oracle_act_info.value)
            oracle_q_buf[t] = oracle_q
            
            assert action.shape == (self.batch_spec.B,) == (len(self.sampling_envs),)
            for b, env in enumerate(self.sampling_envs):
                # Environment inputs and outputs are numpy arrays.
                # o, r, d, env_info = env.step(action[b])
                o, r, terminated, truncated, env_info = env.step(action[b])
                if self.is_atari:
                    self.traj_infos[b].step(
                                    reward=r,
                                    r_hat=None,
                                    raw_reward=env_info["raw_reward"],
                                    terminated=terminated,
                                    truncated=truncated,
                                    need_reset=env.need_reset,
                                    lives=env.ale.lives())
                    if truncated or env.need_reset:
                        if truncated:
                            assert env.need_reset
                        completed_infos.append(self.traj_infos[b].terminate())
                        self.traj_infos[b] = self.TrajInfoCls(total_lives=self.total_env_lives)
                        o, _ = env.reset()
                elif self.is_highway:
                    self.traj_infos[b].step(
                                    reward=r,
                                    terminated=terminated,
                                    truncated=truncated,
                                    info=env_info)
                    if truncated or terminated:
                        completed_infos.append(self.traj_infos[b].terminate())
                        self.traj_infos[b] = self.TrajInfoCls(total_lives=self.total_env_lives)
                        o, _ = env.reset()
                else:
                    raise NotImplementedError
                
                observation[b] = o[:]
                env_buf.reward[t, b] = r
                env_buf.done[t, b] = terminated
                if self.cfg.finetune_cfg.finetune_model.oracle_type == 'hm':
                    self.obs_hm_t[b] = env_info["human_img"][:]
        # self.total_steps += self.batch_spec.T
        self.total_steps += collected_timestep
        return completed_infos, collected_timestep

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
