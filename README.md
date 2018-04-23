

## 环境配置
建议在python3下运行

### Install dependencies

需要另外安装的第三方库主要有：theano keras gym matplotlib

可以参考以下代码进行安装
```pip3 install theano keras gym matplotlib```

### 使用前 
1. 需将环境文件ChannelEnv.pv放到gym的安装目录下。如`/usr/local/lib/python3.6/site-packages/gym/envs/classic_control/`
2. 打开1中目录中的`__init__.py`文件，加入语句： `from gym.envs.classic_control.ChannelEnv import ChannelEnv` 
3. 进入gym安装目录的`gym/gym/envs/`打开`__init__.py`，添加代码： 
```
register( id='Channel-v0',
entry_point='gym.envs.classic_control:ChannelEnv',
max_episode_steps=200, reward_threshold=100.0, )
```

### 完成上述操作后

使用python3运行channel1.py

## Set up environment using Anaconda 

### Install anaconda

see [Conda documentation#Installation](https://conda.io/docs/user-guide/install/index.html) and choose **anaconda3**

### Create Environments
1. Create a conda environment with python 3.6: `conda create --name gymtestbed python=3.6`
2. Switch to created environment: `source <anaconda path>/bin/activate gymtestbed`. For example `source ~anaconda3/bin/activate gymtestbed`.
see also [Conda documentation#Managing Environments](https://conda.io/docs/user-guide/getting-started.html#managing-environments)

### Install dependencies
* See [Conda documentation#Managing packages](https://conda.io/docs/user-guide/getting-started.html#managing-packages) to install `theano`, `keras` and `matplotlib`
* Install `gym` using:
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[classic_control]'
```
see also the github page of [gym](https://github.com/openai/gym).
