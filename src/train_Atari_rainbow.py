#!/usr/bin/env python3
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
from omegaconf import DictConfig

from rlpyt.samplers.collections import BatchSpec
from rlpyt.utils.buffer import buffer_from_example

from old_utils.utils import set_seed_everywhere
from old_utils.logger import Logger, RAINBOW_TRAJ_STATICS,\
                                ATARI_TRAJ_METRICS, HIGHWAY_TRAJ_METRICS,\
                                RAINBOW_CF_TRAJ_STATICS, RAINBOW_CF_TRAJ_METRICS
# from old_utils.replay_buffer import ReplayBuffer
from new_utils.reward_models.preference import PEBBLEAtariRewardModel
from new_utils.rlpyt_utils import delete_ind_from_array, build_samples_buffer, \
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

        if cfg.agent == 'sac_discrete':
            assert cfg.agent.double_q_discrete_critic.encoder.init_type in ['kaiming_linear', 'kaiming_lrelu', 'orthogonal']
            assert cfg.agent.double_q_discrete_critic.encoder.obs_type in ['img', 'rom']
        
        assert cfg.reward_cfg.type_name in ['GT', 'PEBBLEAtari', 'CF']  # GT: ground-truth, PEBBLE: preference
        assert (cfg.agent_save_frequency % cfg.eval_frequency == 0)\
                or cfg.agent_save_frequency < 0

        if cfg.reward_cfg.type_name == 'CF':
            assert cfg.reward_cfg.reward_model.neighbor_size % 2 == 1

        cfg.agent.algo.r_hat_GT_coef = cfg.reward_cfg.r_hat_GT_coef
        cfg.agent.algo.use_potential = cfg.reward_cfg.use_potential
        
        self.work_dir = os.path.abspath(HydraConfig.get().run.dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        cfg.sampler_b = int(cfg.sampler_b)
        cfg.sampler_t = int(cfg.sampler_t)

        set_seed_everywhere(cfg.seed)  # NOTE: this part doesn't set seed for env
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(cfg.device)

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
        self.n_steps = self.cfg.num_env_steps
        self.log_interval_steps = self.cfg.eval_frequency
        self.agent_save_frequency = self.cfg.agent_save_frequency

        # From runner.startup()
        ##   examples = self.sampler.initialize(...)
        if cfg.env.env_name.lower() in atari_env.ATARI_ENV:
            self.is_atari = True
            self.is_highway = False
            self.envs = [atari_env.make_env(cfg) for _ in range(cfg.sampler_b)]
            # env.seed(seed) is removed in gymnasium from v0.26 in favour of env.reset(seed)
            # see https://gymnasium.farama.org/content/migration-guide/#v21-to-v26-migration-guide for more information
            for id_env in range(cfg.sampler_b):
                self.envs[id_env].reset(seed=cfg.seed+id_env)
            
            if cfg.reward_cfg.type_name != 'GT' and cfg.reward_cfg.demo_pretrain.flag:
                raise NotImplementedError  # why did I write this condition?
                # demo_env use the same setting as training env, except the seed
                self.demo_env = atari_env.make_env(cfg)
                self.demo_env.reset(seed=cfg.seed+cfg.sampler_b)
            
            self.eval_envs = [atari_env.make_env(cfg, eval=True) for _ in range(cfg.num_eval_episodes)]
            for id_env in range(cfg.num_eval_episodes):
                self.eval_envs[id_env].reset(seed=cfg.seed+id_env+123321778)

            if cfg.reward_cfg.type_name != 'GT':
                if cfg.reward_cfg.traj_based.flag:
                    self.sampling_env = atari_env.make_env(cfg, eval=True)
                    self.sampling_env.reset(seed=cfg.seed+123456)
            
            self.total_env_lives = self.envs[0].ale.lives()
            print(f'****** original total lives: {self.total_env_lives} ******')
            self.TrajInfoCls = GymAtariTrajInfo
            self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                            #  log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name,
                             reward_type=cfg.reward_cfg.type_name,
                             traj_based=cfg.reward_cfg.traj_based.flag if cfg.reward_cfg.type_name=='CF' else False,
                             env_name='atari')
            if cfg.save_eval_video:
                raise NotImplementedError
        elif cfg.env.env_name in highway_env.HIGHWAY_ENV_NAME:
            self.is_atari = False
            self.is_highway = True
            cfg.env.render_mode = 'rgb_array' if  cfg.env.render_mode=='RGB' else cfg.env.render_mode
            if cfg.save_eval_video:
                cfg.env.render_mode = 'rgb_array'
                self.eval_video_path = os.path.join(self.work_dir, 'evaluation_videos')
                os.makedirs(self.eval_video_path, exist_ok=True)
            self.envs = [highway_env.make_highway_env(cfg) for _ in range(cfg.sampler_b)]
            for id_env in range(cfg.sampler_b):
                self.envs[id_env].reset(seed=cfg.seed+id_env)

            self.eval_envs = [highway_env.make_highway_env(cfg, eval=True) for _ in range(cfg.num_eval_episodes)]
            for id_env in range(cfg.num_eval_episodes):
                self.eval_envs[id_env].reset(seed=cfg.seed+id_env+123321778)
            self.TrajInfoCls = GymHighwayTrajInfo
            self.total_env_lives = 1
            self.logger = Logger(self.work_dir,
                                save_tb=cfg.log_save_tb,
                                #  log_frequency=cfg.log_frequency,
                                agent=cfg.agent.name,
                                reward_type=cfg.reward_cfg.type_name,
                                traj_based=cfg.reward_cfg.traj_based.flag if cfg.reward_cfg.type_name=='CF' else False,
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
        cfg.agent.agent.model.action_dim = int(self.envs[0].action_space.n)
        cfg.agent.agent.model.obs_shape = self.envs[0].observation_space.shape  # (frame_stack, H, W)
        cfg.agent.algo.obs_shape = self.envs[0].observation_space.shape  # (frame_stack, H, W)
        if cfg.agent_model_cfg.encoder_cfg.is_mlp == False:
            cfg.agent_model_cfg.encoder_cfg.in_channels = self.envs[0].observation_space.shape[0]
            cfg.reward_model_cfg.encoder_cfg.in_channels = self.envs[0].observation_space.shape[0]
        self.agent = hydra.utils.instantiate(cfg.agent.agent)
        self.algo = hydra.utils.instantiate(cfg.agent.algo)
        self.agent.initialize(
                              # env_spaces=self.envs[0].spaces,
                              action_dim=self.envs[0].action_space.n,
                            #   share_memory=False,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(
            agent=self.agent,
            env=self.envs[0],
            batch_spec=self.batch_spec,
            agent_shared=False,
            env_shared=False,
            reward_type=cfg.reward_cfg.type_name,
            )
        # NOTE: samples_pyt and samples_np share the same buffer
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        #  Other parts in self.sampler.initialize() are moved to self.run()
        # Back to runner.startup()
        self.itr_batch_size = self.batch_spec.size
        log_interval_itrs = int(max(self.log_interval_steps//self.itr_batch_size, 1))
        n_itr = int(self.n_steps // self.itr_batch_size)
        if n_itr % log_interval_itrs > 0:  # Keep going to next log itr.
            extra_train_itr = log_interval_itrs - (n_itr % log_interval_itrs)
            print(f"WARNING: Train extra {extra_train_itr} itr, \
                  {extra_train_itr*self.itr_batch_size} steps ======")
            n_itr += extra_train_itr
        self.log_interval_itrs = log_interval_itrs
        if cfg.agent_save_frequency > 0:
            self.model_save_interval_itrs = log_interval_itrs * \
                int(self.cfg.agent_save_frequency // self.cfg.eval_frequency)
        else:
            self.model_save_interval_itrs = -1
        self.log_train_avg_itr = int(cfg.log_train_avg_step // self.batch_spec.size)
        self.n_itr = n_itr
        print(f"------ Running {n_itr} iterations of minibatch RL.",
              f"log_interval_itrs: {self.log_interval_itrs},",
              f"model_save_interval_itrs: {self.model_save_interval_itrs},"
              f"log_train_avg_itr: {self.log_train_avg_itr}------")

        self.agent.to_device(self.device)
        if self.is_highway and self.cfg.env.obs.remove_frame_axis == True:
            env_remove_frame_axis = True
        else:
            env_remove_frame_axis = False
        self.algo.initialize(
            agent=self.agent,
            n_itr=self.n_itr,
            batch_spec=self.batch_spec,
            mid_batch_reset=self.mid_batch_reset,
            examples=examples,  # To initialize algo's replay_buffer
            # world_size=self.world_size,
            rank=self.rank,
            remove_frame_axis=env_remove_frame_axis,
        )

        if cfg.reward_cfg.type_name != 'GT':
            assert cfg.reward_cfg.num_interact > cfg.reward_cfg.reward_model.size_segment
        if cfg.reward_cfg.type_name == 'CF':
            assert (cfg.reward_cfg.oracle.exp_path is not None) and \
                    (cfg.reward_cfg.oracle.ckpt_id is not None)
            # in fact, the constraints to buffer size is not so constrainted: just make sure that: when relabel return, the replay buffer is not full, otherwise relabel will be inaccurate
            assert cfg.num_env_steps + cfg.agent.algo.n_step_return < cfg.agent.algo.replay_size  # Otherwise will have issue when relabel return
            
            with open(os.path.join(cfg.reward_cfg.oracle.exp_path,
                                   'exp_logs/metadata.pkl'), 'rb') as f:
                oracle_hydra_cfg = pkl.load(f)
            print(f"oracle.env: {oracle_hydra_cfg['cfg']['env']}")
            print(f"current env: {cfg.env}")
            assert oracle_hydra_cfg['cfg']['env'] == cfg.env
            self.oracle_agent = hydra.utils.instantiate(oracle_hydra_cfg['cfg']['agent']['agent'])
            
            self.oracle_agent.initialize(
                              # env_spaces=self.envs[0].spaces,
                              action_dim=self.envs[0].action_space.n,
                            #   share_memory=False,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
            self.oracle_agent.load(model_dir=os.path.join(
                                        cfg.reward_cfg.oracle.exp_path, 'exp_logs'),
                                    step=cfg.reward_cfg.oracle.ckpt_id,
                                    device=self.device)
            self.step = -1  # to distinguish from normal evaluation, use itr=-1 ans step=-1
            self.logger.log('eval/model_save_duration', -1, self.step)  # stupid hack, to initialize a header in csv
            self.evaluate_agent(itr=-1, agent=self.oracle_agent,\
                                eval_eps=cfg.reward_cfg.oracle.eps)
            # self.oracle_agent.reset()
            self.oracle_agent.eval_mode(itr=1, eps=self.cfg.reward_cfg.oracle.eps)  # NOTE: itr=1 s.t. the oracle also use eps_eval=0.001
        elif (cfg.reward_cfg.type_name == 'PEBBLEAtari') and\
             (cfg.reward_cfg.demo_pretrain.flag == True):
            assert (cfg.reward_cfg.demo_pretrain.expert.exp_path is not None) and \
                    (cfg.reward_cfg.demo_pretrain.expert.ckpt_id is not None)
            assert cfg.num_env_steps + cfg.agent.algo.n_step_return < cfg.agent.algo.replay_size  # Otherwise will have issue when relabel return
            
            with open(os.path.join(cfg.reward_cfg.demo_pretrain.expert.exp_path,
                                   'exp_logs/metadata.pkl'), 'rb') as f:
                demo_hydra_cfg = pkl.load(f)
            print(f"oracle.env: {demo_hydra_cfg['cfg']['env']}")
            print(f"current env: {cfg.env}")
            assert demo_hydra_cfg['cfg']['env'] == cfg.env
            self.demo_agent = hydra.utils.instantiate(demo_hydra_cfg['cfg']['agent']['agent'])
            
            self.demo_agent.initialize(
                              # env_spaces=self.envs[0].spaces,
                              action_dim=self.envs[0].action_space.n,
                            #   share_memory=False,
                              global_B=self.global_B,
                              env_ranks=self.env_ranks)
            self.demo_agent.load(model_dir=os.path.join(
                                cfg.reward_cfg.demo_pretrain.expert.exp_path, 'exp_logs'),
                                step=cfg.reward_cfg.demo_pretrain.expert.ckpt_id,
                                device=self.device)
            self.step = -1  # to distinguish from normal evaluation, use itr=-1 ans step=-1
            self.logger.log('eval/model_save_duration', -1, self.step)  # stupid hack, to initialize a header in csv
            self.evaluate_agent(itr=-1, agent=self.demo_agent,
                                eval_eps=cfg.reward_cfg.demo_pretrain.expert.eps)
            # self.demo_agent.reset()
            self.demo_agent.eval_mode(itr=1, eps=cfg.reward_cfg.demo_pretrain.expert.eps)  # NOTE: itr=1 s.t. the oracle also use eps_eval=0.001

            self.oracle_agent = None
        else:
            self.oracle_agent = None

        self.reward_type = cfg.reward_cfg.type_name
        if cfg.reward_cfg.type_name == 'GT':
            self.reward_model = None
            self.reward_train_finish = True
        elif cfg.reward_cfg.type_name == 'PEBBLEAtari':
            # assert cfg.agent.algo.min_steps_learn < cfg.reward_cfg.num_interact
            # for logging
            self.total_feedback = 0
            self.labeled_feedback = 0
            self.times_train_reward = 0
            # instantiating the reward model
            self.reward_model = hydra.utils.instantiate(cfg.reward_cfg.reward_model)
            self.reward_model.initialize_buffer(
                agent=self.agent,
                env=self.envs[0],
            )
        elif cfg.reward_cfg.type_name == 'CF':
            # assert cfg.agent.algo.min_steps_learn < cfg.reward_cfg.num_interact
            self.total_feedback = 0
            self.labeled_feedback = 0
            self.times_train_reward = 0
            self.reward_model = hydra.utils.instantiate(cfg.reward_cfg.reward_model)
            self.reward_model.initialize_buffer(  # build label buffer for reward model
                agent=self.agent,
                oracle=self.oracle_agent,
                env=self.envs[0],
                check_label_path=os.path.join(self.work_dir, 'CF_labels') \
                                 if self.cfg.reward_cfg.check_label else None,
            )
        else:
            raise NotImplementedError
        
        if self.reward_model is not None:
            os.makedirs(os.path.join(self.work_dir, 'init'), exist_ok=True)
            self.reward_model.save(os.path.join(self.work_dir, 'init'),
                                   title='init')
            if cfg.reward_cfg.reward_model.reset.flag:
                if cfg.reward_cfg.reward_model.reset.type == 'init':
                    if cfg.reward_cfg.reward_model.ckpt_path is None:
                        self.reward_model.load_init(
                            os.path.join(self.work_dir, 'init'),
                            title='init')
                    else:  
                        self.reward_model.load_init(
                            os.path.join(cfg.reward_cfg.reward_model.reset.init_ckpt_path),
                            title='init')
                        
            if (cfg.reward_cfg.use_potential.ahead or cfg.reward_cfg.use_potential.back) \
                and cfg.reward_cfg.use_potential.agent_potential:
                if cfg.reward_cfg.use_potential.back:
                    raise NotImplementedError  # didn't implement this in agent.py
                self.agent.config_potential(potential_model=self.reward_model)
            
            assert cfg.reward_cfg.seed_step <= cfg.agent.algo.min_steps_learn
            if cfg.reward_cfg.early_advertising.flag:  # although this flag should not be set to 'True'
                assert cfg.reward_cfg.max_feedback >= cfg.reward_cfg.seed_step
            self.reward_seed_itr = int(cfg.reward_cfg.seed_step // self.itr_batch_size)
            self.reward_train_finish = False

        os.makedirs(self.work_dir, exist_ok=True)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))  # save cfg file

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
    
        self.logger.log('eval/duration', eval_duration, self.step)
        if agent is None:
            self._cum_eval_time += eval_duration
        
        # steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        # logger.record_tabular('StepsInEval', steps_in_eval)
        self.logger.log('eval/num_completed_traj', len(eval_completed_traj_infos), self.step)

        true_episode_reward_list = [info["Return"] for info in eval_completed_traj_infos]
        episode_reward_hat_list = [info["ReturnRHat"] for info in eval_completed_traj_infos]
        true_raw_episode_reward_list = [info["RawReturn"] for info in eval_completed_traj_infos]
        episode_length_list = [info["Length"] for info in eval_completed_traj_infos]

        for lst_name, stat_name in itertools.product(
                        ATARI_TRAJ_METRICS.keys(), RAINBOW_TRAJ_STATICS.keys()):
            self.logger.log(f'eval/{lst_name}_{stat_name}',
                             eval(f'np.{stat_name}({lst_name}_list)'), self.step)
        
        self.logger.log('eval/eps_this_eval', eps_this_eval, self.step)
        self.logger.log('eval/iter', itr, self.step)
        self.logger.dump(self.step, ty='eval')

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
        
        # If agent is not None, this function is used to evaluate oracle_Agent for CF reward
        eval_agent = agent if agent else self.agent
        eval_traj_infos = [
            self.TrajInfoCls(total_lives=self.total_env_lives) for _ in range(len(self.eval_envs))]
        completed_traj_infos = []
        observations = []
        for env in self.eval_envs:
            observations.append(env.reset()[0][:])
        observation = buffer_from_example(observations[0], len(self.eval_envs))
        for b, o in enumerate(observations):
            observation[b] = o
        obs_pyt = torchify_buffer((observation))  # shape: (#eval_envs, C*frame_stack, H, W)

        # eval_agent.reset()
        eps_this_eval = eval_agent.eval_mode(itr=(1 if agent else itr),
                                             eps=eval_eps)  # itr > 1 will use eval_eps (or eps is specified), =0 will be random agent
        if (agent is None) and self.reward_model:
            self.reward_model.eval()

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

            if (agent is None) and self.reward_model:  # When evaluate the loaded oracle, we pass the oracle agent here
                r_hat_sa = self.reward_model.r_hat_sa(obs_pyt, act_pyt)  # r_hat_sa_pyt.shape: (B,)
            else:
                r_hat_sa = [None for _ in range(len(live_envs))]

            b = 0
            while b < len(live_envs):  # don't want to do a for loop since live envs changes over time
                env_id = live_envs[b]
                next_o, r, terminated, truncated, env_info = self.eval_envs[env_id].step(action[b])
                # traj_infos[env_id].step(r, terminated)
                eval_traj_infos[env_id].step(
                                reward=r,
                                r_hat=r_hat_sa[b],
                                raw_reward=env_info["raw_reward"],
                                terminated=terminated,
                                truncated=truncated,
                                need_reset=env.need_reset,
                                lives=env.ale.lives())
                if truncated or self.eval_envs[env_id].need_reset:
                    completed_traj_infos.append(eval_traj_infos[env_id].terminate())

                    observation = delete_ind_from_array(observation, b)
                    action = delete_ind_from_array(action, b)
                    r_hat_sa = delete_ind_from_array(r_hat_sa, b)

                    obs_pyt = torchify_buffer((observation))
                    act_pyt = torchify_buffer((action))
                    # r_hat_sa_pyt = torchify_buffer((r_hat_sa))

                    del live_envs[b]
                    b -= 1  # live_envs[b] is now the next env, so go back one.
                else:
                    observation[b] = next_o[:]

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

                    return completed_traj_infos, eps_this_eval
            eval_t += 1

        if eval_t >= self.cfg.max_frames:
            print(f"!!!!!WARNING:Evaluation reached max num time steps {self.cfg.max_frames},",
                  f" but still have {len(live_envs)} env.!!!!!")
        
        return completed_traj_infos, eps_this_eval
    
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
    
        self.logger.log('eval/duration', eval_duration, self.step)
        if agent is None:
            self._cum_eval_time += eval_duration
        
        # steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        # logger.record_tabular('StepsInEval', steps_in_eval)
        self.logger.log('eval/num_completed_traj', len(eval_completed_traj_infos), self.step)
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
                             eval(f'np.{stat_name}({lst_name}_list)'), self.step)
        
        self.logger.log('eval/eps_this_eval', eps_this_eval, self.step)
        self.logger.log('eval/iter', itr, self.step)
        self.logger.dump(self.step, ty='eval')
    
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
        
        if self.cfg.save_eval_video:
            video_path = os.path.join(self.eval_video_path, itr)
            os.makedirs(video_path, exist_ok=True)
            raise NotImplementedError  # haven't write the video writer part, the obs_img can be captured from returned info['human_img']
        
        # If agent is not None, this function is used to evaluate oracle_Agent for CF reward
        eval_agent = agent if agent else self.agent
        eval_traj_infos = [
            self.TrajInfoCls(total_lives=self.total_env_lives) for _ in range(len(self.eval_envs))]
        completed_traj_infos = []
        observations = []
        for env in self.eval_envs:
            observations.append(env.reset()[0][:])
        observation = buffer_from_example(observations[0], len(self.eval_envs))
        for b, o in enumerate(observations):
            observation[b] = o
        obs_pyt = torchify_buffer((observation))  # shape: (#eval_envs,D)

        # eval_agent.reset()
        eps_this_eval = eval_agent.eval_mode(itr=(1 if agent else itr),
                                             eps=eval_eps)  # itr > 1 will use eval_eps (or eps is specified), =0 will be random agent
        if (agent is None) and self.reward_model:
            self.reward_model.eval()

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

            if (agent is None) and self.reward_model:  # When evaluate the loaded oracle, we pass the oracle agent here
                r_hat_sa = self.reward_model.r_hat_sa(obs_pyt, act_pyt)  # r_hat_sa_pyt.shape: (B,)
            else:
                r_hat_sa = [None for _ in range(len(live_envs))]

            b = 0
            while b < len(live_envs):  # don't want to do a for loop since live envs changes over time
                env_id = live_envs[b]
                next_o, r, terminated, truncated, env_info = self.eval_envs[env_id].step(action[b])
                # traj_infos[env_id].step(r, terminated)
                eval_traj_infos[env_id].step(
                                reward=r,
                                terminated=terminated,
                                truncated=truncated,
                                info=env_info)
                if truncated or terminated:
                    completed_traj_infos.append(eval_traj_infos[env_id].terminate())

                    observation = delete_ind_from_array(observation, b)
                    action = delete_ind_from_array(action, b)
                    r_hat_sa = delete_ind_from_array(r_hat_sa, b)

                    obs_pyt = torchify_buffer((observation))
                    act_pyt = torchify_buffer((action))
                    # r_hat_sa_pyt = torchify_buffer((r_hat_sa))

                    del live_envs[b]
                    b -= 1  # live_envs[b] is now the next env, so go back one.
                else:
                    observation[b] = next_o[:]

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

                    return completed_traj_infos, eps_this_eval
            eval_t += 1

        if eval_t >= self.cfg.max_frames:
            print(f"!!!!!WARNING:Evaluation reached max num time steps {self.cfg.max_frames},",
                  f" but still have {len(live_envs)} env.!!!!!")
        
        return completed_traj_infos, eps_this_eval

    def learn_reward(self, first_flag=False, early_advertising=False, demo_pretrain=False,
                     traj_based=False):
        # get feedbacks (get_queries -> get_label -> put_queries)
        sample_st = time.time()
        if traj_based:
            labeled_queries = self.reward_model.traj_based_sampling(
                                                    agent=self.agent,
                                                    oracle_agent=self.oracle_agent,
                                                    env=self.sampling_env,
                                                    TrajInfoCls=self.TrajInfoCls,
                                                    fb_ratio=self.cfg.reward_cfg.traj_based.fb_ratio,
                                                    fb_per_traj=self.cfg.reward_cfg.traj_based.fb_per_traj,
                                                    )
        elif early_advertising:
            labeled_queries = self.reward_model.early_advertising_states_in_buffer()
            assert labeled_queries == self.reward_model.len_inputs_T
        elif demo_pretrain:
            labeled_queries = self.reward_model.pretrain_sampling(
                num_feedback=self.cfg.reward_cfg.demo_pretrain.num_feedback)
        elif first_flag:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.reward_cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.reward_cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.reward_cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.reward_cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.reward_cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.reward_cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        sample_duration = time.time() - sample_st
        # TODO: consider the relationship between sampling and reset: what's the order of the 2 steps? Can reset be used for sampling?
        #       For now, we first use previous reward model to do sampling, then we reset and train
        if self.cfg.reward_cfg.type_name == 'CF':
            if early_advertising:
                self.total_feedback += labeled_queries
            elif traj_based:
                self.total_feedback += labeled_queries
            elif demo_pretrain:
                raise NotImplementedError
            else:
                self.total_feedback += self.reward_model.query_batch * self.reward_model.cf_per_seg
        elif self.cfg.reward_cfg.type_name == 'PEBBLEAtari':
            # For preference reward model,some segments may be deprecated (e.g. similar return between two segments)
            if early_advertising:
                raise NotImplementedError
            elif traj_based:  # TODO: For Pref, maybe one fair comparison with CF-traj_based is to sample segments from evaluation instead of training data?
                raise NotImplementedError
            elif demo_pretrain:
                self.total_feedback += labeled_queries
            else:
                self.total_feedback += self.reward_model.query_batch
        else:
            raise NotImplementedError
        if not traj_based:
            self.reward_train_finish = (self.total_feedback >= self.cfg.reward_cfg.max_feedback)
        self.labeled_feedback += labeled_queries  # NOTE: not all raised queries are available, some queries (e.g. two segments with similar total reward) are deprecated
        
        train_acc = 0
        if self.labeled_feedback > 0:
            self.reward_model.train()
            # update reward
            if traj_based:
                if first_flag:
                    reward_update = self.cfg.reward_cfg.traj_based.reward_update_itr0
                else:
                    reward_update = self.cfg.reward_cfg.traj_based.reward_update
            elif early_advertising:
                reward_update = self.cfg.reward_cfg.early_advertising.early_update
            elif demo_pretrain:
                reward_update = self.cfg.reward_cfg.demo_pretrain.reward_update
            else:
                reward_update = self.cfg.reward_cfg.reward_update
            
            if (not first_flag) and self.cfg.reward_cfg.reward_model.reset.flag:
                self.reward_model.model_reset()
            for epoch in range(reward_update):
                if self.cfg.reward_cfg.type_name == 'CF':
                    train_acc, train_losses, train_grad_norms = self.reward_model.train_reward()
                elif self.cfg.reward_cfg.type_name == 'PEBBLEAtari':
                    if self.cfg.reward_cfg.reward_model.label_margin > 0 \
                        or self.cfg.reward_cfg.reward_model.teacher_eps_equal > 0:
                        train_acc, train_losses, train_grad_norms = self.reward_model.train_soft_reward()
                    else:
                        train_acc, train_losses, train_grad_norms = self.reward_model.train_reward()
                else:
                    raise NotImplementedError

                self.logger.log('interact_acc/epoch', epoch, self.step)
                for id_en, acc_en in enumerate(train_acc):
                    # self.logger.log('interact_acc/id_ensemble', id_en, self.step)
                    self.logger.log(f'interact_acc/action_acc_ensemble_{id_en}', acc_en, self.step)
                    self.logger.log(f'interact_acc/loss_ensemble_{id_en}', train_losses[id_en], self.step)
                    self.logger.log(f'interact_acc/grad_norm_ensemble_{id_en}', train_grad_norms[id_en], self.step)
                
                total_acc = np.mean(train_acc)  # average over ensembles
                self.logger.log('interact_acc/action_acc_avg', total_acc, self.step)
                self.logger.log('interact_acc/action_acc_max', np.max(train_acc), self.step)
                self.logger.log('interact_acc/action_acc_min', np.min(train_acc), self.step)
                self.logger.log('interact_acc/action_acc_std', np.std(train_acc), self.step)
                
                total_loss = np.mean(train_losses)
                self.logger.log('interact_acc/loss_avg', total_loss, self.step)
                self.logger.log('interact_acc/loss_max', np.max(train_losses), self.step)
                self.logger.log('interact_acc/loss_min', np.min(train_losses), self.step)
                self.logger.log('interact_acc/loss_std', np.std(train_losses), self.step)

                self.logger.dump(self.step, ty='interact_acc')
                
                if early_advertising:
                    target_acc = self.cfg.reward_cfg.early_advertising.early_acc_target
                elif demo_pretrain:
                    target_acc = self.cfg.reward_cfg.demo_pretrain.target_acc
                else:
                    target_acc = self.cfg.reward_cfg.acc_target
                if total_acc > target_acc:
                    break
            
            self.logger.log('interact/reward_update', epoch+1, self.step)
            self.logger.log('interact/final_action_acc', total_acc, self.step)
            self.logger.log('interact/final_action_acc_max', np.max(train_acc), self.step)
            self.logger.log('interact/final_loss', total_loss, self.step)
            self.logger.log('interact/final_loss_min', np.min(train_losses), self.step)
            print(f"[R_HAT] Reward function in step {self.step} is updated!! ACC: " + str(total_acc))
        else:
            self.logger.log('interact/reward_update', 0, self.step)
            print(f"[R_HAT] No labeled feedback to train reward function in step {self.step}")
        
        return labeled_queries, sample_duration
    
    def run(self):
        # From SerialSampler.initialize()
        # agent_inputs, traj_infos = collector.start_envs(self.max_decorrelation_steps=0)
        self.traj_infos = [self.TrajInfoCls(total_lives=self.total_env_lives) for _ in range(len(self.envs))]
        
        obs_ls = list()
        for env in self.envs:
            obs_ls.append(env.reset()[0][:])
        observation = buffer_from_example(obs_ls[0], len(self.envs))
        for b, obs in enumerate(obs_ls):
            observation[b] = obs  # numpy array or namedarraytuple
        self.agent_inputs = observation
        
        self.agent.collector_initialize(
            global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
            env_ranks=self.env_ranks,
        )
        # self.agent.reset()
        self.agent.sample_mode(itr=0)
        
        # From 
        self.step = 0
        self.iter = 0
        self.min_itr_learn = self.algo.min_itr_learn
        self._cum_eval_time = 0
        self._cum_completed_trajs = 0
        global_start_time = time.time()
        interact_count = 0
        first_flag = True  # To learn reward model, for the first training update, query should be sampled uniformly

        if self.cfg.reward_cfg.traj_based.flag:
            assert self.reward_seed_itr == 0  # because traj_based does not need to collect an initial buffer
        # n_itr = self.startup()
        # self.n_itr = n_itr
        for itr in range(self.n_itr):
            self.itr = itr
            # evaluation
            if itr > 0 and self.model_save_interval_itrs > 0 and \
                           itr % self.model_save_interval_itrs == 0:
                save_model_start = time.time()
                self.agent.save(self.work_dir, self.step)
                # if self.reward_model is not None:
                #     self.reward_model.save(self.work_dir, self.step)
                save_model_end = time.time()
                self.logger.log('eval/model_save_duration', save_model_end-save_model_start, self.step)
            if itr % self.log_interval_itrs == 0:
                if itr == 0 and not self.cfg.eval_at_itr0:
                    pass
                else:
                    if not (itr > 0 and self.model_save_interval_itrs > 0
                             and itr % self.model_save_interval_itrs == 0):
                        self.logger.log('eval/model_save_duration', -1, self.step)
                    self.evaluate_agent(itr=itr,
                        eval_eps=(self.cfg.agent.agent.eps_eval if self.cfg.agent.agent.ckpt_path else None))
            
            # train reward model
            if self.reward_model is not None:
                if itr == self.reward_seed_itr:
                    allow_update = True
                elif self.cfg.reward_cfg.traj_based.flag:
                    allow_update = (self.reward_model.total_traj < self.cfg.reward_cfg.traj_based.max_num_traj)\
                                    and (interact_count == self.cfg.reward_cfg.num_interact)
                else:
                    allow_update = (self.total_feedback < self.cfg.reward_cfg.max_feedback) and\
                                    (interact_count == self.cfg.reward_cfg.num_interact)
                if allow_update:
                    # The first time to train reward needs to collect data during #reward_seed_itr 
                    # After the 1st time, we train the reward model according to reward_cfg.num_interact
                    if (itr < self.reward_seed_itr) and\
                        (interact_count == self.cfg.reward_cfg.num_interact):
                        # The 1st time to train the reward model follows reward_seed_itr instead of num_interact
                        interact_count = 0
                    else:
                        # (itr == self.min_itr_learn): start to train
                        # or
                        # (self.total_feedback < self.cfg.reward_cfg.max_feedback) 
                        #     and (interact_count == self.cfg.reward_cfg.num_interact):
                        #     reach the interact period
                        # update schedule
                        if itr == self.reward_seed_itr:
                            # set a special query_batch for the 1st phase
                            early_advertising = self.cfg.reward_cfg.early_advertising.flag
                            demo_pretrain = self.cfg.reward_cfg.demo_pretrain.flag
                            traj_based = self.cfg.reward_cfg.traj_based.flag
                            assert not (early_advertising and demo_pretrain)  # how to combine the two cases? hot to count feedback if combine the two?
                            assert not (early_advertising and traj_based)  # strange, have't consider this case
                            assert not (demo_pretrain and traj_based)  # strange, have't consider this case
                            if early_advertising:
                                self.reward_model.set_batch(0)
                            elif demo_pretrain:
                                self.reward_model.set_batch(self.cfg.reward_cfg.demo_pretrain.num_feedback)
                                self.reward_model.add_demo_data(
                                    agent=self.demo_agent,
                                    agent_eps=self.cfg.reward_cfg.demo_pretrain.expert.eps,
                                    env=self.demo_env,
                                )
                            elif traj_based:
                                self.reward_model.set_traj(self.cfg.reward_cfg.traj_based.num_traj_0)
                                assert not ((self.cfg.reward_cfg.traj_based.fb_ratio is not None) and\
                                            (self.cfg.reward_cfg.traj_based.fb_per_traj is not None))  # can only use one of the config
                                assert not ((self.cfg.reward_cfg.traj_based.fb_ratio is None) and\
                                            (self.cfg.reward_cfg.traj_based.fb_per_traj is None))  # must have one of the value
                            else:
                                self.reward_model.set_batch(self.cfg.reward_cfg.query_batch_0)
                        else:
                            if self.cfg.reward_cfg.reward_schedule == 1:  # more steps, smaller frac
                                frac = (self.n_steps - self.step) / self.n_steps
                                if frac == 0:
                                    frac = 0.01
                            elif self.cfg.reward_cfg.reward_schedule == 2:  # more steps, larger frac
                                frac = self.n_steps / (self.n_steps - self.step + 1)
                            elif self.cfg.reward_cfg.reward_schedule == 0:
                                frac = 1
                            else:
                                raise NotImplementedError
                            if not traj_based:
                                self.reward_model.change_batch(frac)
                            else:
                                self.reward_model.change_traj(frac)
                            early_advertising = False
                            demo_pretrain = False
                        
                        # corner case: new total feed > max feed
                        if traj_based:
                            if self.reward_model.total_traj + self.reward_model.query_traj > self.cfg.reward_cfg.traj_based.max_num_traj:
                                self.reward_model.set_traj(self.cfg.reward_cfg.traj_based.max_num_traj - self.reward_model.total_traj)
                        else:
                            if (not early_advertising) and \
                                (self.reward_model.query_batch + self.total_feedback > self.cfg.reward_cfg.max_feedback):
                                self.reward_model.set_batch(self.cfg.reward_cfg.max_feedback - self.total_feedback)
                        
                        # # update margin --> not necessary / will be updated soon
                        # new_margin = np.mean(self.reward_model.avg_train_step_return) * self.cfg.reward_cfg.reward_model.size_segment
                        # # TODO: I think PEBBLE's original set_teacher_thres_skip is wrong, which equals to new_margin * self.cfg.teacher_eps_skip**2
                        # # self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        # # self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        # self.reward_model.set_teacher_thres_skip(new_margin)
                        # self.reward_model.set_teacher_thres_equal(new_margin)

                        start_time = time.time()
                        label_this_itr, sample_duration = self.learn_reward(
                                                           first_flag=first_flag, early_advertising=early_advertising,
                                                           demo_pretrain=demo_pretrain, traj_based=traj_based)
                        self.times_train_reward += 1
                        first_flag = False
                        end_time = time.time()
                        self.logger.log('interact/sample_duration', sample_duration, self.step)
                        self.logger.log('interact/learn_duration', end_time-start_time, self.step)

                        if demo_pretrain:
                            save_dir = os.path.join(self.work_dir, "demo_pretrain")
                            os.makedirs(save_dir, exist_ok=True)
                            self.reward_model.save(save_dir, f"fb{self.total_feedback}_t{self.step}")
                        if early_advertising:
                            save_dir = os.path.join(self.work_dir, "early_advertising")
                            os.makedirs(save_dir, exist_ok=True)
                            self.reward_model.save(save_dir, f"fb{self.total_feedback}_t{self.step}")
                        if not (demo_pretrain or early_advertising):
                            if (self.cfg.reward_cfg.reward_save_interval is not None) and\
                                (self.times_train_reward % self.cfg.reward_cfg.reward_save_interval == 0):
                                if traj_based:
                                    self.reward_model.save(save_dir, f"traj{self.reward_model.total_traj}_t{self.step}")
                                else:
                                    self.reward_model.save(save_dir, f"fb{self.total_feedback}_t{self.step}")

                        start_time = time.time()
                        self.reward_model.eval()
                        if label_this_itr > 0:
                            self.algo.replay_buffer.relabel_with_predictor(self.reward_model)
                        end_time = time.time()
                        self.logger.log('interact/relabel_duration', end_time-start_time, self.step)
                        
                        # one feedback may not be counted as one label. E.g., render two segments but the expert think it's useless to compare the two
                        self.logger.log('interact/feedback_count', self.total_feedback, self.step)
                        if self.cfg.reward_cfg.type_name == 'CF':
                            if traj_based:
                                self.logger.log('interact/traj_this_itr', self.reward_model.query_traj, self.step)
                                assert self.reward_model.query_traj == len(self.reward_model.completed_infos)
                                self.logger.log('interact/total_traj', self.reward_model.total_traj, self.step)
                                
                                true_episode_reward_list = [info["Return"] for info in self.reward_model.completed_infos]
                                true_raw_episode_reward_list = [info["RawReturn"] for info in self.reward_model.completed_infos]
                                episode_length_list = [info["Length"] for info in self.reward_model.completed_infos]
                                for lst_name, stat_name in itertools.product(
                                                RAINBOW_CF_TRAJ_METRICS.keys(), RAINBOW_CF_TRAJ_STATICS.keys()):
                                    self.logger.log(f'interact/{lst_name}_{stat_name}',
                                                    eval(f'np.{stat_name}({lst_name}_list)'), self.step)
                                self.logger.log('interact/feedback_traj_avg', label_this_itr/self.reward_model.query_traj, self.step)
                                self.logger.log('interact/feedback_this_itr', label_this_itr, self.step)
                            else:
                                self.logger.log('interact/feedback_this_itr',
                                                self.reward_model.len_inputs_T if early_advertising else\
                                                    self.reward_model.query_batch * self.reward_model.cf_per_seg,
                                                self.step)
                        elif self.cfg.reward_cfg.type_name == 'PEBBLEAtari':
                            if traj_based:
                                raise NotImplementedError
                            else:
                                self.logger.log('interact/feedback_this_itr',
                                                self.reward_model.query_batch,
                                                self.step)
                        else:
                            raise NotImplementedError
                        self.logger.log('interact/label_count', self.labeled_feedback, self.step)
                        self.logger.log('interact/label_this_itr', label_this_itr, self.step)
                        self.logger.log('interact/len_inputs_T', self.reward_model.len_inputs_T, self.step)
                        self.logger.log('interact/len_label', self.reward_model.len_label, self.step)
                        self.logger.dump(self.step, ty='interact')
                        interact_count = 0
            
            itr_train_start_time = time.time()
            completed_infos = self.collect_batch(itr)
            interact_count += self.batch_spec.size

            if self.samples_pyt is not None:  # filled with collected interaction from self.collect_batch()
                samples_to_buffer = self.algo.samples_to_buffer(self.samples_pyt,
                                        use_r_hat=(self.reward_model is not None))
                self.algo.replay_buffer.append_samples(samples_to_buffer)
                
                if (self.reward_model is not None) and (not self.reward_train_finish) \
                    and (not self.cfg.reward_cfg.traj_based.flag):
                    if self.oracle_agent is not None:  # corrective
                        # oracle_act, oracle_act_prob = self.oracle.act(obs, return_prob=True)  # NOTE: oracle_act_prob is well-defined probability (sum to 1) since the last layer in actor is a softmax
                        # self.reward_model.add_data(obs, action, oracle_act, oracle_act_prob, terminated)
                        samples_to_reward_buffer = self.reward_model.samples_to_reward_buffer(
                                                                    self.samples_pyt)
                        self.reward_model.add_data(samples_to_reward_buffer)
                    else:  # Preference
                        # adding data (with ground-truth reward) to the reward training data
                        # self.reward_model.add_data(obs, action, reward, terminated)
                        samples_to_reward_buffer = self.reward_model.samples_to_reward_buffer(
                                                                    self.samples_pyt)
                        self.reward_model.add_data(samples_to_reward_buffer)

            self.agent.train_mode(itr)
            # opt_info = self.algo.optimize_agent(itr, self.samples_pyt)
            opt_info = self.algo.optimize_agent(itr)
            
            self.step = (itr + 1) * self.batch_spec.size
            self._cum_completed_trajs += len(completed_infos)
            itr_train_end_time = time.time()
            # TODO: maybe also log some information from training env
            self.logger.log('train/duration', itr_train_end_time - itr_train_start_time, self.step,
                            repeat_count=self.log_train_avg_itr)
            self.logger.log('train/total_duration', itr_train_end_time - global_start_time, self.step,
                            repeat_count=self.log_train_avg_itr)
            self.logger.log('train/loss', np.average(opt_info.loss), self.step,
                            repeat_count=self.log_train_avg_itr)
            self.logger.log('train/gradNorm', np.average(opt_info.gradNorm), self.step,
                            repeat_count=self.log_train_avg_itr)
            self.logger.log('train/tdAbsErr', np.average(opt_info.tdAbsErr), self.step,
                            repeat_count=self.log_train_avg_itr)
            if (itr + 1) % self.log_train_avg_itr == 0:
                self.logger.log('train/iter', itr+1, self.step)
                self.logger.dump(self.step, ty='train')
        
        global_end_time = time.time() 
        print(f'[Total Running Time]: {global_end_time - global_start_time}')

        if self.model_save_interval_itrs > 0:
            self.agent.save(self.work_dir, self.step)
            # if self.reward_model is not None:
            #     self.reward_model.save(self.work_dir, self.step)
        
        final_end_time = time.time()
        self.logger.log('eval/model_save_duration', final_end_time - global_end_time, self.step)

        self.evaluate_agent(itr=self.n_itr)

    def collect_batch(self, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env  # leading_dims: [T, B]
        if self.reward_model:
            r_hat_buf = self.samples_np.r_hat
            if self.oracle_agent:
                oracle_act_buf = self.samples_np.oracle_act
                oracle_act_prob_buf = self.samples_np.oracle_act_prob
                oracle_q_buf = self.samples_np.oracle_q
            self.reward_model.eval()
        completed_infos = list()
        # observation, action, reward = agent_inputs
        observation = self.agent_inputs  # observation.shape = (B, frame_stack, H, W)
        # obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        obs_pyt = torchify_buffer(self.agent_inputs)  # share self.agent_inputs's memory
        # agent_buf.prev_action[0] = action  # Leading prev_action.
        # env_buf.prev_reward[0] = reward
        if itr==0 and self.agent.ckpt_path is not None:
            # If we have already has a ckpt, we can directly use it to generate high-quality data
            self.agent.eval_mode(itr, eps=0.01)
        else:
            self.agent.sample_mode(itr)
        
        for t in range(self.batch_spec.T):
            env_buf.observation[t] = observation  # slice 
            # NOTE: "env_buf.observation[t]" will not be modified after "observation" being updated
            # Agent inputs and outputs are torch tensors.
            # act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            act_pyt, agent_info = self.agent.step(obs_pyt)  # NOTE: no_grad for agent.step
            if self.reward_model:
                r_hat_sa = self.reward_model.r_hat_sa(obs_pyt, act_pyt)  # r_hat_sa.shape: (B,)
                if self.oracle_agent:
                    oracle_act_pyt, oracle_act_info = self.oracle_agent.step(obs_pyt)
                    # oracle_act_pyt.shape: (B,), oracle_act_info.p.shape: (B, action_dim)

            action = numpify_buffer(act_pyt)  # shape (B,)
            if self.reward_model:
                r_hat_buf[t] = r_hat_sa  # NOTE: changing this buffer will change samples_np & samples_pyt at the same time
                if self.oracle_agent:
                    oracle_act = numpify_buffer(oracle_act_pyt)
                    oracle_act_buf[t] = oracle_act

                    oracle_act_prob_pyt = F.softmax(
                        oracle_act_info.value / self.cfg.reward_cfg.reward_model.softmax_tau,
                        dim=-1)
                    oracle_act_prob = numpify_buffer(oracle_act_prob_pyt)
                    oracle_act_prob_buf[t] = oracle_act_prob

                    oracle_q = numpify_buffer(oracle_act_info.value)
                    oracle_q_buf[t] = oracle_q
            
            assert action.shape == (self.batch_spec.B,) == (len(self.envs),)
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                # o, r, d, env_info = env.step(action[b])
                o, r, terminated, truncated, env_info = env.step(action[b])
                if self.is_atari:
                    self.traj_infos[b].step(
                                    reward=r,
                                    r_hat=r_hat_sa[b] if self.reward_model else None,
                                    raw_reward=env_info["raw_reward"],
                                    terminated=terminated,
                                    truncated=truncated,
                                    need_reset=env.need_reset,
                                    lives=env.ale.lives())
                    # if getattr(env_info, "traj_done", d):
                    if truncated or env.need_reset:
                        if truncated:
                            assert env.need_reset
                        completed_infos.append(self.traj_infos[b].terminate())
                        self.traj_infos[b] = self.TrajInfoCls(total_lives=self.total_env_lives)
                        o, _ = env.reset()
                else:
                    assert self.is_highway
                    self.traj_infos[b].step(
                                    reward=r,
                                    terminated=terminated,
                                    truncated=truncated,
                                    info=env_info)
                    # if getattr(env_info, "traj_done", d):
                    if truncated or terminated:
                        completed_infos.append(self.traj_infos[b].terminate())
                        self.traj_infos[b] = self.TrajInfoCls(total_lives=self.total_env_lives)
                        o, _ = env.reset()
                # if terminated:
                #     self.agent.reset_one(idx=b)
                observation[b] = o[:]
                # reward[b] = r
                env_buf.reward[t, b] = r
                # env_buf.done[t, b] = d
                env_buf.done[t, b] = terminated
                # if env_info:
                #     env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            # env_buf.reward[t] = reward
            # if agent_info:
            #     agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            raise NotImplementedError
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        # return AgentInputs(observation, action, reward), traj_infos, completed_infos
        return completed_infos

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
