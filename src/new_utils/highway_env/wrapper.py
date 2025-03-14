import gymnasium as gym
# from gym.wrappers import FlattenObservation
import pdb
import copy
import numpy as np
import pprint
import itertools

# from gymnasium.wrappers import FrameStack, FlattenObservation
from gymnasium.error import ResetNeeded
from gymnasium import spaces

import highway_env
import highway_env.utils as hiutils
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.abstract import Observation
from typing import Dict, Text, Tuple, Optional


HIGHWAY_ENV_NAME = {
    'highway': 'highway-v0',
    'highwayFast': 'highway-fast-v0',
}

HIGHWAY_ENV_OBS = {
    'GI': {
        'type': 'GrayscaleObservation',
    },
    'KM': {
        'type': 'Kinematics',
        'features': ["presence", "x", "y", "vx", "vy"],
    },
}

HIGHWAY_ENV_ACT = {
    'DMeta': {
        'type': 'DiscreteMetaAction',
    },
    'Disc': {
        'type': 'DiscreteAction',
    },
}


class HighwayWrapper(gym.Wrapper):  # better to use wrapper, instead of inheriting from HighwayEnv or HighwayEnvFast
    def __init__(self, env: gym.Env, reward_cfg, obs_type):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self.config.update(reward_cfg)
        if self.config["normalize_reward"]:
            self.reward_items = [
                "collision_reward",
                "non_collision_reward",
                "right_lane_reward",
                "high_speed_reward",
                "low_speed_reward",
                "lane_change_reward",
            ]
            self.reward_min = np.sum([self.config[name] for name in self.reward_items if self.config[name] < 0])
            self.reward_max = np.sum([self.config[name] for name in self.reward_items if self.config[name] > 0])
        
        self.obs_type = obs_type
        if obs_type == 'Kinematics':
            self.observation_space = spaces.flatten_space(env.observation_space)
            self.KM_flatten = True
        else:
            self.KM_flatten = False
        
        if type(self.env.unwrapped.action_type) == highway_env.envs.common.action.DiscreteAction:
            acceleration_arr = np.linspace(-1., 1., self.env.unwrapped.action_type.actions_per_axis)
            steering_arr = np.linspace(-1., 1., self.env.unwrapped.action_type.actions_per_axis)
            self.action_name_val = [f"acc{acc}_ste{ste}" for acc, ste in itertools.product(acceleration_arr, steering_arr)]
        
        self.save_human_img = False
    
    def _reward(self, lane_changed: bool) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(lane_changed)
        reward = sum(self.config[name] * reward \
                        for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            if self.config["normalize_reward_neg"]:
                reward = hiutils.lmap(reward,
                                    [self.reward_min, self.reward_max],
                                    [-1, 1])
            else:
                reward = hiutils.lmap(reward,
                                    [self.reward_min, self.reward_max],
                                    [0, 1])
        reward *= float(self.vehicle.on_road)
        return reward

    def _rewards(self, lane_changed: bool) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)  # all lanes, including the current ego-lane
        lane = self._get_lane()
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = hiutils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "non_collision_reward": 1.0 - float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),  # -1 because lane in range [0, len(neighbours)-1]
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "low_speed_reward": float(self.vehicle.speed < self.config["reward_speed_range"][0]),
            # "on_road": float(self.vehicle.on_road),
            "lane_change_reward": lane_changed,
        }

    def _get_lane(self):
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
                    else self.vehicle.lane_index[2]
        return lane
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if not self.env._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        old_lane_id = self._get_lane()
        self.time += 1 / self.config["policy_frequency"]  # policy_frequency [Hz]: the number of decisions in 1s
        self.total_timesteps += 1
        self.env.unwrapped._simulate(action)
        new_lane_id = self._get_lane()
        lane_changed = (new_lane_id != old_lane_id)
        
        obs = self.observation_type.observe()
        reward = self._reward(lane_changed=lane_changed)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = self._info(action=action, lane_changed=lane_changed)
        # NOTE: in highway, we can calculate the distance from forward_speed, since the road is a straight line without any rotation
        self.total_distance += info["forward_speed"] * 1 / self.config["policy_frequency"]
        self.cnt_lane_changed += int(lane_changed)
        self.total_off_road_time += (1 - info["on_road"]) * 1 / self.config["policy_frequency"]
        self.total_collision_reward += info["rewards"]["collision_reward"]
        self.total_non_collision_reward += info["rewards"]["non_collision_reward"]
        self.total_right_lane_reward += info["rewards"]["right_lane_reward"]
        self.total_high_speed_reward += info["rewards"]["high_speed_reward"]
        self.total_low_speed_reward += info["rewards"]["low_speed_reward"]
        # self.total_lane_change_reward += info["rewards"]["lane_change_reward"]
        info["total_distance"] = self.total_distance
        info["cnt_lane_changed"] = self.cnt_lane_changed
        info["total_off_road_time"] = self.total_off_road_time

        info["total_collision_reward"] = self.total_collision_reward
        info["total_non_collision_reward"] = self.total_non_collision_reward
        info["total_right_lane_reward"] = self.total_right_lane_reward
        info["total_high_speed_reward"] = self.total_high_speed_reward
        info["total_low_speed_reward"] = self.total_low_speed_reward
        # info["total_lane_change_reward"] = self.total_lane_change_reward
        
        if self.render_mode == 'human':
            self.render()
        elif self.render_mode == 'rgb_array':
            human_img = self.render()
            info['human_img'] = human_img

        if self.KM_flatten:
            obs = spaces.flatten(self.env.observation_space, obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        if self.KM_flatten:
            obs = spaces.flatten(self.env.observation_space, obs)  # obs.shape: (#vehicles * #features,)
        self.total_distance = 0  # [m]
        self.cnt_lane_changed = 0
        self.total_off_road_time = 0  # [s]
        self.total_collision_reward = 0
        self.total_non_collision_reward = 0
        self.total_right_lane_reward = 0
        self.total_high_speed_reward = 0
        self.total_low_speed_reward = 0
        # self.total_lane_change_reward = 0
        self.time = 0
        self.total_timesteps = 0
        info = self._info(action=self.action_space.sample(),
                          lane_changed=False)
        info["total_distance"] = self.total_distance
        info["cnt_lane_changed"] = self.cnt_lane_changed
        info["total_off_road_time"] = self.total_off_road_time

        info["total_collision_reward"] = self.total_collision_reward
        info["total_non_collision_reward"] = self.total_non_collision_reward
        info["total_right_lane_reward"] = self.total_right_lane_reward
        info["total_high_speed_reward"] = self.total_high_speed_reward
        info["total_low_speed_reward"] = self.total_low_speed_reward
        # info["total_lane_change_reward"] = self.total_lane_change_reward

        if self.render_mode == 'human':
            self.render()
        elif self.render_mode == 'rgb_array':
            human_img = self.render()
            info['human_img'] = human_img
        return obs, info
    
    def _info(self, action: Action, lane_changed: bool) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        if type(self.env.unwrapped.action_type) == highway_env.envs.common.action.DiscreteAction:
            action_name = self.action_name_val[action]
        elif type(self.env.unwrapped.action_type) == highway_env.envs.common.action.DiscreteMetaAction:
            action_name = self.env.unwrapped.action_type.actions[action]
        else:
            raise NotImplementedError
        info = {
            "speed": self.vehicle.speed,
            "forward_speed": self.vehicle.speed * np.cos(self.vehicle.heading),
            "crashed": self.vehicle.crashed,
            "on_road": float(self.vehicle.on_road),
            "new_lane_id": self._get_lane(),
            "time": self.time,  # [s]
            "action": action,
            "action_name": action_name,
            "total_timesteps": self.total_timesteps,
        }
        info["rewards"] = self._rewards(lane_changed=lane_changed)
        return info
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


def make_highway_env(cfg, eval=False):
    # NOTE: do not change the order of wrappers, otherwise must check terminated/truncated/need_reset again carefully.
    env = gym.make(HIGHWAY_ENV_NAME[cfg.env.env_name],
                   render_mode=cfg.env.render_mode)
    obs_cfg = copy.deepcopy(HIGHWAY_ENV_OBS[cfg.env.obs.type])
    act_cfg = copy.deepcopy(HIGHWAY_ENV_ACT[cfg.env.act.type])
    reward_cfg = copy.deepcopy(cfg.env.reward)
    reward_cfg["reward_speed_range"] = [
            float(reward_cfg["reward_speed_range"]["min"]),
            float(reward_cfg["reward_speed_range"]["max"]),
        ]

    if cfg.env.obs.type == 'GI':  # gray image
        obs_cfg['observation_shape'] = (cfg.env.obs.shape.w, cfg.env.obs.shape.h)
        obs_cfg['stack_size'] = cfg.env.obs.stack_size
        # obs.shape: (stack, W, H)
        assert cfg.env.obs.shape.w == cfg.env.obs.shape.h  # I haven't check if Atari use HW will conflict with WH
        pdb.set_trace()  # check if there has some other configs
    elif cfg.env.obs.type == 'KM':  # Kinematics
        obs_cfg['vehicles_count'] = cfg.env.obs.vehicles_count
        for k, v in cfg.env.obs.features.items():
            if v == True:
                obs_cfg['features'].append(k)
    else:
        raise NotImplementedError
    
    if cfg.env.act.type == 'DMeta':
        assert cfg.env.act.actions_per_axis == None
        if 'target_speeds' in cfg.env.act.keys():
            act_cfg['target_speeds'] = np.linspace(
                        cfg.env.act['target_speeds']['st'],
                        cfg.env.act['target_speeds']['ed'],
                        cfg.env.act['target_speeds']['cnt'])
            # print(f"act_cfg['target_speeds']:{act_cfg['target_speeds']}")
    elif cfg.env.act.type == 'Disc':
        assert type(cfg.env.act.actions_per_axis) == int
        assert cfg.env.act.actions_per_axis in [3, 5, 7, 9]  # since available control values are obtained with np.linspace(low, high, actions_per_axis), we need to use those values to make sure 0 in the candidates and values are symmetric.
        act_cfg['actions_per_axis'] = cfg.env.act.actions_per_axis
    else:
        raise NotImplementedError

    env.config.update({  # do not need to update reward_cfg since we will handel it in HighwayWrapper
            'observation': obs_cfg,
            'action': act_cfg,
            'lanes_count': cfg.env.lanes_count,
            'vehicles_count': cfg.env.vehicles_count,
            'duration': cfg.env.duration,
            'show_trajectories': cfg.env.show_traj,
            'vehicles_density': cfg.env.vehicles_density,
            'policy_frequency': cfg.env.policy_frequency,
            'simulation_frequency': cfg.env.simulation_frequency,
        })
    
    obs_before_wrap, info_before_wrap = env.reset()
    obs_shape_before_wrap = env.observation_space.shape  # KM: (#vehicle, #features). dtype: np.float32. (-np.inf, np.inf); GI: (c, w, h)
    
    wrapped_env = HighwayWrapper(env, 
                                reward_cfg=reward_cfg,
                                obs_type=obs_cfg["type"])
    obs_wrapped, info_wrapped = wrapped_env.reset()
    
    if cfg.env.obs.type == 'KM':
        assert obs_wrapped.shape == (obs_shape_before_wrap[0] * obs_shape_before_wrap[1],)  # (f, d)
    else:
        pdb.set_trace()  # check shape for wrapped_env
        pdb.set_trace()  # add a reshape wrapper? e.g. flatten for KM, CWH->CHW for GI
        assert cfg.env.obs.shape.w == cfg.env.obs.shape.h  # I haven't check if Atari use HW will conflict with WH
        assert obs_wrapped.shape == (cfg.env.obs.stack_size,
                                       cfg.env.obs.shape.w,
                                       cfg.env.obs.shape.w)
    print(f'-------[ENV:{cfg.env.env_name}] action_type: {wrapped_env.action_type}')
    print(f'-------[ENV:{cfg.env.env_name}] action_space: {wrapped_env.action_space}')
    print(f'-------[ENV:{cfg.env.env_name}] observation_type: {wrapped_env.observation_type}')
    print(f'-------[ENV:{cfg.env.env_name}] observation_space: {wrapped_env.observation_space}')
    return wrapped_env