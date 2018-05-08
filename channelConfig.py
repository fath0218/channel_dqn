## ChannelEnv
CORRECT_REWARD = 30.0  #wrong to correct
CR_TO_CR_REWARD = -10.0  #correct to another correct
PUNISH_REWARD = -80.0  #wrong to itself
WR_TO_WR_REWARD = -50.0  #wrong to another wrong
CR_TO_WR_REWARD = -100.0  #correct to wrong
STUBBORN_REWARD = 10.0 #stick to a certain correct channel



CHANNEL_CNT = 10
STATE_CNT = CHANNEL_CNT * 2   #double of channal count

BLOCK_CNT = 3
USR_CNT = 3
REFRESH = 1
REFRESH_METHOD_OLD = 0

##channel1
literation = 500
