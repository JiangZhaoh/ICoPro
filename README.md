# RLHF_CF
conda create -n bpref python==3.9
# package install
install pytorch from https://pytorch.org/
pip install hydra-core --upgrade
pip install tensorboard termcolor
pip install "gymnasium[atari]"
pip install gym matplotlib
pip install gymnasium[accept-rom-license]
pip install gymnasium[other]
pip install highway-env
<!-- pip install tkvideoplayer -->
## Package for User Interface
### MacOS
pip install wxPython
### Ubuntu 22.04
<!-- Check https://wxpython.org/pages/downloads/index.html for other OS -->
pip install -U \
    -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04/ \
    wxPython
## download rlpyt
cd ./packages
git clone https://github.com/astooke/rlpyt.git
cd rlpyt
pip install -e .

# ICoPro (Rainbow + Corrective Feedback + Proxy Reward)
python src/finetune_RLRND_rainbow_Q.py --config-name finetune_RLRND_rainbow_Q finetune_cfg=CF_RLRND_Rainbow
        finetune_cfg.finetune_model.RL_loss.flag=True finetune_cfg.finetune_model.RL_loss.human_label.sl_weight=0.5 finetune_cfg.finetune_model.RL_loss.tgt_label.sl_weight=0.5 finetune_cfg.use_ft=True
        finetune_cfg.oracle.exp_path=[path for expert checkpoint] finetune_cfg.oracle.ckpt_id=[checkpoint id]
## For Atari, append the following commends for ICoPro's
    env.env_name=frostbite
    num_env_steps=2000000 finetune_cfg.steps_per_itr=5000 finetune_cfg.finetune_model.query_batch=20
## For highway, Append the following commends for ICoPro's
    env=highwayKM agent_model_cfg=highwayMLP env.reward.lane_change_reward=0.2 env.reward.right_lane_reward=0 env.reward.high_speed_reward=1.5 env.reward.low_speed_reward=-0.5 env.reward.collision_reward=-1.7
    num_env_steps=150000 finetune_cfg.steps_per_itr=1000 finetune_cfg.finetune_model.query_batch=10 finetune_cfg.finetune_model.RL_loss.RL_recent_itr=100 finetune_cfg.finetune_model.size_segment=10

## User Study for ICoPro, append the following commends for ICoPro's
finetune_cfg.finetune_model.oracle_type=hm eval_at_itr0=False finetune_cfg.eps_greedy.flag=True finetune_cfg.eps_greedy.eps_itr=1 agent_save_freq=1 save_eval_video.flag=True 
### For Atari, append the following commends further
save_eval_video.shape_h=210 save_eval_video.shape_w=160
### For highway, append the following commends further
env.render_mode=rgb_array 

# To Obtain Expert Checkpoint (Rainbow + Proxy Reward)
python train_Atari_rainbow.py --config-name train_atari_rainbow num_env_steps=2000000 eval_frequency=100000 agent.algo.replay_size=100000 agent.algo.min_steps_learn=5000 agent.algo.eps_steps=5001 num_eval_episodes=1
## For Atari, append the following commends
    env.env_name=freeway
## For Highway,  append the following commends
    env=highway_KM agent_model_cfg=highwayMLP

# Ablations
Append the following commends for ICoPro's
## PVP Setting
finetune_cfg.finetune_model.RL_loss.flag=True  finetune_cfg.finetune_model.RL_loss.tgt_label.sl_weight=0 finetune_cfg.finetune_model.RL_loss.human_label.sl_weight=1 finetune_cfg.use_ft=False finetune_cfg.finetune_model.loss_name=PV finetune_cfg.finetune_model.RL_loss.use_reward=False

## RLIF Setting
finetune_cfg.use_ft=False
finetune_cfg.finetune_model.RL_loss.flag=True
finetune_cfg.finetune_model.RL_loss.use_RLIF_reward=True
finetune_cfg.finetune_model.RL_loss.human_label.sl_weight=0
finetune_cfg.finetune_model.RL_loss.tgt_label.sl_weight=0
