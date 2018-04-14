使用前 
1 需将环境文件ChannelEnv.pv放到gym的安装目录下。如/usr/local/lib/python3.6/site-packages/gym/envs/classic_control/ 
2 打开1中目录中的__init__.py文件，加入语句： from gym.envs.classic_control.ChannelEnv import ChannelEnv 
3 进入gym安装目录的gym/gym/envs/打开__init__.py，添加代码： register( id='Channel-v0',
entry_point='gym.envs.classic_control:ChannelEnv',
max_episode_steps=200, reward_threshold=100.0, )

完成上述操作后
使用python3运行channel1.py
