import numpy as np
import pdb

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack, FlattenObservation, TimeLimit

try:
    import cv2
except ImportError:
    cv2 = None

ATARI_ENV = {
    # use environment that frameskip=1 and repeat_action_probability=0(v4)
    # TODO: shall we consider repeat_action_probability=0.25(v0)?
    'pong': {'ram': 'Pong-ramNoFrameskip-v4',  # lives=0, #action=6
             'img': 'PongNoFrameskip-v4'},
    'freeway': {'ram': 'Freeway-ramNoFrameskip-v4',  # lives=0, #action=3
             'img': 'FreewayNoFrameskip-v4'},
    'battlezone': {'ram': 'BattleZone-ramNoFrameskip-v4',  # lives=5, #action=18
             'img': 'BattleZoneNoFrameskip-v4'},
    'seaquest': {'ram': 'Seaquest-ramNoFrameskip-v4',  # lives=4, #action=18
             'img': 'SeaquestNoFrameskip-v4'},
    'frostbite': {'ram': 'Frostbite-ramNoFrameskip-v4',  # lives=4, #action=18
             'img': 'FrostbiteNoFrameskip-v4'},
    'mspacman': {'ram': 'MsPacman-ramNoFrameskip-v4',  # lives=3, #action=9
             'img': 'MsPacmanNoFrameskip-v4'},
    'boxing': {'ram': 'Boxing-ramNoFrameskip-v4',  # lives=0, #action=18
             'img': 'BoxingNoFrameskip-v4'},
    'breakout': {'ram': 'Breakout-ramNoFrameskip-v4',  # lives=5, #action=4
             'img': 'BreakoutNoFrameskip-v4'},
    'mr': {'img': 'MontezumaRevengeNoFrameskip-v4'},  # lives=6, #action=18
    'enduro': {'img': 'EnduroNoFrameskip-v4'},  # lives=0, #action=9
    'alien': {'img': 'AlienNoFrameskip-v4'},  # lives=?, #action=18
    'choppercommand': {'img': 'ChopperCommandNoFrameskip-v4'},  # lives=?, #action=18
    'hero': {'img': 'HeroNoFrameskip-v4'},  # lives=?, #action=18
    'spaceinvaders': {'img': 'SpaceInvadersNoFrameskip-v4'},  # lives=?, #action=6
}

ENV_SCORE_AREA = {  # (x, y, h, w) to denote the score region, (x, y) is the left upper corner
    # NOTE: for img_size = (84 * 84)
    'pong': [(0, 0, 84, 9)],
    'freeway': [(0, 0, 84, 5)],
    'breakout': [(0, 0, 45, 6)],
}

class AtariPreprocessing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Atari 2600 preprocessing wrapper.
    This code is modified from gymnasium.wrappers.AtariPreprocessing

    This class follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".

    Specifically, the following preprocess stages applies to the atari environment:

    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default
    - Max-pooling: Pools over the most recent two observations from the frame skips
    - Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
        Turned off by default. Not recommended by Machado et al. (2018).
    - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default
    - Grayscale observation: If the observation is colour or greyscale, by default, greyscale.
    - Scale observation: If to scale the observation between [0, 1) or [0, 255), by default, not scaled.
    """

    def __init__(
        self,
        env: gym.Env,
        clip_reward: bool,
        obs_type: str,
        log_reward: bool = False,
        noop_max: int = 30,
        frame_skip: int = 4,
        terminal_on_life_loss: bool = False,
        screen_size: int = 84,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
        fire_on_reset=False,
    ):
        """ Wrapper for Atari 2600 preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            screen_size (int): resize Atari frame
            terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `terminated=True` whenever a
                life is lost.
            grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
                is returned.
            grayscale_newaxis (bool): `if True and grayscale_obs=True`, then a channel axis is added to
                grayscale observations to make them 3-dimensional.
            scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
                optimization benefits of FrameStack Wrapper.

        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        assert scale_obs == False  # handle scaling in agent part, in order to save memory
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale_obs,
            grayscale_newaxis=grayscale_newaxis,
            scale_obs=scale_obs,
        )
        gym.Wrapper.__init__(self, env)

        if cv2 is None:
            raise gym.error.DependencyNotInstalled(
                "opencv-python package not installed, run `pip install gymnasium[other]` to get dependencies for atari"
            )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            if (
                env.spec is not None
                and "NoFrameskip" not in env.spec.id
                and getattr(env.unwrapped, "_frameskip", None) != 1
            ):
                raise ValueError(
                    "Disable frame-skipping in the original env. Otherwise, more than one "
                    "frame-skip will happen as through this wrapper"
                )

        self._has_fire = "FIRE" in env.unwrapped.get_action_meanings() and fire_on_reset
        if self._has_fire:
            # same assertation as stable-baseline3
            assert env.unwrapped.get_action_meanings()[1] == "FIRE"
            print(f'****** Fire on Reset ******')

        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        self.action_names = env.unwrapped.get_action_meanings()
        print(f'****** Total {len(env.unwrapped.get_action_meanings())} Action Meanings ******')
        for id, a_name in enumerate(env.unwrapped.get_action_meanings()):
            print(f'*** {id:2d} {a_name}')

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs
        self.clip_reward = clip_reward
        self.log_reward = log_reward
        assert not (self.clip_reward and self.log_reward)

        self.ram_obs = (obs_type == 'rom')  # TODO: fix this part
        if self.ram_obs:
            assert not self.scale_obs
        else:
            assert obs_type == 'img'  # grey scale image

        # buffer of most recent two observations for max pooling
        assert isinstance(env.observation_space, Box)

        if self.ram_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]
        elif self.grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]
        self.human_img_buffer = [
            np.empty(env.observation_space.shape, dtype=np.uint8),
            np.empty(env.observation_space.shape, dtype=np.uint8),
        ]

        self.lives = 0
        # self.game_over = False
        self.need_reset = False

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        if self.ram_obs:
            _shape = (128,)
        else:
            _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
            if grayscale_obs and not grayscale_newaxis:
                _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

        self.unwrapped.save_human_img = None

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale

    def step(self, action):
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            # self.game_over = terminated
            # NOTE: we use self.need_reset to indicate algorithm if the env need to call env.reset()
            #       Some times the env returns a terminated but do not need to be reset in the cases that it is terminated due to life loss
            self.need_reset = terminated or truncated  # need reset

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                terminated = terminated or new_lives < self.lives
                # self.game_over = terminated
                if new_lives < self.lives and self._has_fire:
                    self.env.step(1)
                self.lives = new_lives

            if terminated or truncated:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[1])
                if self.unwrapped.save_human_img:
                    self.ale.getScreenRGB(self.human_img_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[0])
                if self.unwrapped.save_human_img:
                    self.ale.getScreenRGB(self.human_img_buffer[0])
                self.ale.getScreenRGB(self.human_img_buffer[0])

        info["raw_reward"] = total_reward
        # total_reward = np.sign(total_reward) if self.clip_reward else total_reward
        if self.clip_reward:
            total_reward = np.sign(total_reward)
        elif self.log_reward:
            total_reward = np.sign(total_reward) * np.log(1 + np.abs(total_reward))

        if self.unwrapped.save_human_img:
            info["human_img"] = self._get_human_obs()

        # if truncated:
        #     print(f'------ reach time limit with lives = {self.ale.lives()} ------')
        return self._get_obs(), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment using preprocessing."""
        # NoopReset
        _, reset_info = self.env.reset(**kwargs)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)  # TODO: how to generate reproducible results with np_random?
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                _, reset_info = self.env.reset(**kwargs)
        
        if self._has_fire:
            self.env.step(1)
        
        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)

        if self.unwrapped.save_human_img:
            self.ale.getScreenRGB(self.human_img_buffer[0])
            self.human_img_buffer[1].fill(0)
            reset_info["human_img"] = self._get_human_obs()  # (H, W, C)
        
        self.need_reset = False
        return self._get_obs(), reset_info

    def _get_human_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.human_img_buffer[0], self.human_img_buffer[1],
                       out=self.human_img_buffer[0])
        assert cv2 is not None
        obs = self.human_img_buffer[0][...]
        obs = np.asarray(obs, dtype=np.uint8)

        return obs  # (H, W, C)
    
    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        assert cv2 is not None
        obs = cv2.resize(
            self.obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs

def make_env(cfg, eval=False):
    # NOTE: do not change the order of wrappers, otherwise must check terminated/truncated/need_reset again carefully.
    env = gym.make(ATARI_ENV[cfg.env.env_name][cfg.env.obs_type])
    # TimeLimit before AtariPreprocessing because AtariPreprocessing have frame_skip, but we set TimeLimit based on frames
    # FrameStack after AtariPreprocessing because FrameStack need t stack processed frames
    wrapped_env = FrameStack(AtariPreprocessing(TimeLimit(env,
                                                          max_episode_steps=cfg.env.max_frames),
                                                obs_type=cfg.env.obs_type,
                                                clip_reward=True if eval else cfg.env.clip_reward,
                                                log_reward=False if eval else cfg.env.log_reward,
                                                terminal_on_life_loss=True if eval else cfg.env.terminal_on_life_loss,  # note that in atari, terminated != need_reset, since some env has more than 1 lives
                                                fire_on_reset=cfg.env.fire_on_reset),
                             num_stack=cfg.env.frame_stack)  # stacked_shape: (#frame, original_shape)
    if cfg.env.obs_type == 'rom':
        pdb.set_trace()  # I think flatten should be ahead of framestack?
        wrapped_env = FlattenObservation(wrapped_env)
    # TODO: replace the score area with a constant black background on all games.
    #       On BeamRider we additionally blank out the enemy ship count,
    #       and on Enduro we blank out the speedometer.
    # TODO: consider how to deal with episodic end（no variable-length episode during training）
    return wrapped_env

def mask_img_score_func_(env_name, obs):
    assert len(obs.shape) in [4, 5]  # (B, C, H, W) for CF; (B, size_segment, C, H, W) for preference
    assert obs.shape[-1] == obs.shape[-2] == 84
    area_list = ENV_SCORE_AREA[env_name]  # score area
    # mask_obs = copy.deepcopy(obs)
    for area in area_list:
        x, y, h, w = area
        obs[..., x:x+w, y:y+h] = 0
    # from new_utils.draw_utils import ax_plot_img
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax_plot_img(ori_img=obs[0, -1], ax=ax, vmin=0, vmax=1)
    # fig.savefig(fname=f'./test_mask.png')

def add_gaussian_noise(img, amplitude):
    # img.dtype == dtype('uint8')
    # img data range: [0, 255]
    # type(img): np.ndarray
    noise = amplitude * np.random.normal(size=img.shape)
    noisy_img = np.clip(img+noise, a_min=0., a_max=1.)
    return noisy_img
