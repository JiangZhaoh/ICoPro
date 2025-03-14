import numpy as np
import pdb
import copy

from rlpyt.samplers.collections import AttrDict
from rlpyt.utils.buffer import buffer_from_example
# from rlpyt.samplers.collections import Samples, AgentSamples, EnvSamples
from rlpyt.utils.collections import namedarraytuple

from new_utils.new_agent.rainbow_agent import AgentInputs
from new_utils.tensor_utils import torchify_buffer


Samples = namedarraytuple("Samples", ["agent", "env"])
SamplesRHatCF = namedarraytuple("SamplesRHat", ["agent", "env", "r_hat",
                                "oracle_act", "oracle_act_prob", "oracle_q"])
SamplesFinetuneCF = namedarraytuple("SamplesRHat", ["agent", "env",
                                "oracle_act", "oracle_act_prob", "oracle_q"])
SamplesRHatPEBBLE = namedarraytuple("SamplesRHat", ["agent", "env", "r_hat"])
AgentSamples = namedarraytuple("AgentSamples",
    ["action"])
    # ["action", "agent_info"])
EnvSamples = namedarraytuple("EnvSamples",
    # ["observation", "reward", "done"])
    ["observation", "reward", "done", "human_img"])


def delete_ind_from_array(array, ind):
    tensor = np.concatenate([array[:ind], array[ind+1:]], 0)
    return tensor


class GymAtariTrajInfo(AttrDict):
    def __init__(self, total_lives, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.ReturnRHat = 0
        self.RawReturn = 0
        self.Episodic_info = {
            "terminated": [False,],
            "truncated": [False,],
            "need_reset": [False,],
            "lives": [total_lives],
            "steps": [0],
        }

    def step(self, reward, r_hat, raw_reward, terminated, truncated, need_reset, lives):
        self.Length += 1
        self.Return += reward
        self.ReturnRHat += r_hat if (r_hat is not None) else 0
        self.RawReturn += raw_reward
        if terminated or truncated or need_reset:
            self.Episodic_info["terminated"].append(terminated)
            self.Episodic_info["truncated"].append(truncated)
            self.Episodic_info["need_reset"].append(need_reset)
            self.Episodic_info["lives"].append(lives)
            self.Episodic_info["steps"].append(self.Length)

    def terminate(self):
        return self
    

class GymHighwayTrajInfo(AttrDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.cnt_step = 0
        self.total_reward = 0
        self.average_speed = 0.
        self.average_forward_speed = 0.
        self.Episodic_info = {}

    def step(self, reward, terminated, truncated, info):
        self.cnt_step += 1
        self.total_reward += reward
        self.average_speed += info["speed"]
        self.average_forward_speed += info["forward_speed"]
        if terminated or truncated:
            self.Episodic_info["terminated"] = terminated
            self.Episodic_info["truncated"] = truncated
            self.Episodic_info["cnt_step"] = self.cnt_step
            self.Episodic_info["crashed"] = info["crashed"]
            self.Episodic_info["time"] = info["time"]
            self.Episodic_info["total_distance"] = info["total_distance"]
            self.Episodic_info["cnt_lane_changed"] = info["cnt_lane_changed"]
            self.Episodic_info["total_off_road_time"] = info["total_off_road_time"]
            self.Episodic_info["total_collision_reward"] = info["total_collision_reward"]
            self.Episodic_info["total_non_collision_reward"] = info["total_non_collision_reward"]
            self.Episodic_info["total_right_lane_reward"] = info["total_right_lane_reward"]
            self.Episodic_info["total_high_speed_reward"] = info["total_high_speed_reward"]
            self.Episodic_info["total_low_speed_reward"] = info["total_low_speed_reward"]
            # self.Episodic_info["total_lane_change_reward"] = info["total_lane_change_reward"]

    def terminate(self):
        self.average_speed /= self.cnt_step
        self.average_forward_speed /= self.cnt_step
        return self


def build_samples_buffer(
                        agent,
                        env,
                        batch_spec,
                        # bootstrap_value,  # NOTE: rlpyt seems only =True for policy gradient methods
                        agent_shared,
                        env_shared,
                        reward_type,
                        ):
    examples = dict()
    # From get_example_outputs(agent, env, examples)
    env.reset()
    a = env.action_space.sample()
    o, r, terminated, truncated, env_info = env.step(a)
    r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
    # agent.reset()
    agent_inputs = torchify_buffer(o)  # agent_inputs.shape: (C*frame_stack, H, W)
    agent.eval_mode(itr=-1)
    a, agent_info = agent.step(agent_inputs)  # a.shape: (), agent_info.value.shape: (action_dim,)
    examples["observation"] = o[:]
    examples["reward"] = r
    examples["done"] = terminated
    examples["human_img"] = env_info.get("human_img", None)
    # examples["truncated"] = truncated
    # examples["env_info"] = env_info
    examples["action"] = a  # OK to put torch tensor here, could numpify.
    # examples["agent_info"] = agent_info

    T, B = batch_spec
    action = buffer_from_example(examples["action"], (T, B), agent_shared)  # action.shape: (T, B)
    # action = all_action[1:]
    # prev_action = all_action[:-1]  # Writing to action will populate prev_action.
    # agent_info = buffer_from_example(examples["agent_info"], (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        # prev_action=prev_action,
        # agent_info=agent_info,
    )
    # if bootstrap_value:
    #     bv = buffer_from_example(examples["agent_info"].value, (1, B), agent_shared)
    #     agent_buffer = AgentSamplesBsv(*agent_buffer, bootstrap_value=bv)

    observation = buffer_from_example(examples["observation"], (T, B), env_shared)  # observation.shape: (T, B, C*frame_Stack, H, W), dtype: uint8
    reward = buffer_from_example(examples["reward"], (T, B), env_shared)  # reward.shape: (T, B), float32
    # reward = all_reward[1:]
    # prev_reward = all_reward[:-1]  # Writing to reward will populate prev_reward.
    done = buffer_from_example(examples["done"], (T, B), env_shared)  # done.shape: (T, B), bool
    # env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
    human_img = buffer_from_example(examples["human_img"], (T, B), env_shared)  # done.shape: (T, B), bool
    env_buffer = EnvSamples(
        observation=observation,  # highway: KM: (T, B, #vehicle*#feature)
        reward=reward,  # (T, B)
        # prev_reward=prev_reward,
        done=done,  # (T, B)
        human_img=human_img,
    )

    if reward_type == 'CF':
        reward_hat_buffer = buffer_from_example(examples["reward"], (T, B), env_shared)  # shape: (T, B), float32
        oracle_act_buffer = buffer_from_example(examples["action"], (T, B), env_shared)  # shape: (T, B), int64
        oracle_act_prob_buffer = buffer_from_example(copy.deepcopy(agent_info.value),  # use deepcopy to avoid using same buffer with oracle_q
                                                     (T, B), env_shared)  # shape: (T, B, action_dim), float32
        oracle_act_q_buffer = buffer_from_example(agent_info.value, (T, B), env_shared)  # shape: (T, B, action_dim), float32
        samples_np = SamplesRHatCF(agent=agent_buffer, env=env_buffer,
                                    r_hat=reward_hat_buffer,
                                    oracle_act=oracle_act_buffer,
                                    oracle_act_prob=oracle_act_prob_buffer,
                                    oracle_q=oracle_act_q_buffer)
    elif reward_type == 'CF_ft':
        oracle_act_buffer = buffer_from_example(examples["action"], (T, B), env_shared)  # shape: (T, B), int64
        oracle_act_prob_buffer = buffer_from_example(copy.deepcopy(agent_info.value),  # use deepcopy to avoid using same buffer with oracle_q
                                                     (T, B), env_shared)  # shape: (T, B, action_dim), float32
        oracle_act_q_buffer = buffer_from_example(agent_info.value, (T, B), env_shared)  # shape: (T, B, action_dim), float32
        samples_np = SamplesFinetuneCF(agent=agent_buffer, env=env_buffer,
                                    oracle_act=oracle_act_buffer,
                                    oracle_act_prob=oracle_act_prob_buffer,
                                    oracle_q=oracle_act_q_buffer)
    elif reward_type == 'CF_RLft':  # same with CF_ft
        oracle_act_buffer = buffer_from_example(examples["action"], (T, B), env_shared)  # shape: (T, B), int64
        oracle_act_prob_buffer = buffer_from_example(copy.deepcopy(agent_info.value),  # use deepcopy to avoid using same buffer with oracle_q
                                                     (T, B), env_shared)  # shape: (T, B, action_dim), float32
        oracle_act_q_buffer = buffer_from_example(agent_info.value, (T, B), env_shared)  # shape: (T, B, action_dim), float32
        samples_np = SamplesFinetuneCF(agent=agent_buffer, env=env_buffer,
                                    oracle_act=oracle_act_buffer,
                                    oracle_act_prob=oracle_act_prob_buffer,
                                    oracle_q=oracle_act_q_buffer)
    elif reward_type == 'PEBBLEAtari':
        reward_hat_buffer = buffer_from_example(examples["reward"], (T, B), env_shared)
        samples_np = SamplesRHatPEBBLE(agent=agent_buffer, env=env_buffer,
                                        r_hat=reward_hat_buffer)
    elif reward_type == 'GT':
        samples_np = Samples(agent=agent_buffer, env=env_buffer)
    else:
        raise NotImplementedError
    samples_pyt = torchify_buffer(samples_np)  # samples_np and samples_pyt share the same buffer
    
    return samples_pyt, samples_np, examples
