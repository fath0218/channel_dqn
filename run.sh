#!/bin/bash
# Update the environment, then run the agent
cp ChannelEnv.py ../gym/gym/envs/classic_control/ChannelEnv.py
python ./channel1.py
