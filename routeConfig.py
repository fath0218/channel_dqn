## ChannelEnv
CORRECT_REWARD = 30.0  #wrong to correct
CR_TO_CR_REWARD = 5.0  #correct to another correct
PUNISH_REWARD = -80.0  #wrong to itself
WR_TO_WR_REWARD = -50.0  #wrong to another wrong
CR_TO_WR_REWARD = -100.0  #correct to wrong
STUBBORN_REWARD = 10.0 #stick to a certain correct channel

#CHANNEL_CNT = 10
#STATE_CNT = CHANNEL_CNT * 2   #double of channal count

PATH_A_CNT = 3
PATH_B_CNT = 4
PATH_C_CNT = 4
TOTAL_PATH_CNT = 7
STATE_A_CNT = PATH_A_CNT * 3
STATE_B_CNT = PATH_B_CNT * 3
STATE_C_CNT = PATH_C_CNT * 3 #available blocked occupied


USR_CNT = 3
REFRESH = 1
REFRESH_METHOD_OLD = 0

## Jammer
BLOCK_CNT = 3 # number of channel jammed at each time slot
#JAMMER_TYPE = 'Random_jammer_1'
JAMMER_TYPE = 'Markov_jammer'
MARKOV_CONFIG = 2

# jamming probability
CHANNEL_P = [[0.2, 0.5, 0.8, 0.5, 0.2, 0.05, 0.02], [0.02, 0.2, 0.5, 0.8, 0.5, 0.2, 0.05], [0.02, 0.05, 0.2, 0.5, 0.8, 0.5, 0.2]]
CHANNEL_P1 = [0.2, 0.5, 0.8, 0.5, 0.2, 0.05, 0.02]

##channel1
literation = 10000

OBSERV_BATCH = 3 #number of timeslot to be observed
