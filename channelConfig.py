## ChannelEnv
CORRECT_REWARD = 30.0  #wrong to correct
CR_TO_CR_REWARD = 5.0  #correct to another correct
PUNISH_REWARD = -80.0  #wrong to itself
WR_TO_WR_REWARD = -50.0  #wrong to another wrong
CR_TO_WR_REWARD = -100.0  #correct to wrong
STUBBORN_REWARD = 10.0 #stick to a certain correct channel

CHANNEL_CNT = 10
STATE_CNT = CHANNEL_CNT * 2   #double of channal count

USR_CNT = 3
REFRESH = 1
REFRESH_METHOD_OLD = 0

## Jammer
BLOCK_CNT = 3 # number of channel jammed at each time slot
#JAMMER_TYPE = 'Random_jammer_1'
#JAMMER_TYPE = 'Random_jammer_2'
JAMMER_TYPE = 'Markov_jammer'
# jamming probability
CHANNEL_P = [[0.22, 0.21, 0.05, 0.02, 0.25, 0.14, 0.02, 0.01, 0.04, 0.04], [0, 0, 0.15, 0.15, 0, 0, 0.15, 0.25, 0.15, 0.15]]

##channel1
literation = 2000

OBSERV_BATCH = 2 #number of timeslot to be observed
