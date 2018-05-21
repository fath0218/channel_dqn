import math
import gym
import logging
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import routeConfig as routeConfig

logger = logging.getLogger(__name__)

CORRECT_REWARD = routeConfig.CORRECT_REWARD  #wrong to correct
CR_TO_CR_REWARD = routeConfig.CR_TO_CR_REWARD  #correct to another correct
PUNISH_REWARD = routeConfig.PUNISH_REWARD  #wrong to itself
WR_TO_WR_REWARD = routeConfig.WR_TO_WR_REWARD  #wrong to another wrong
CR_TO_WR_REWARD = routeConfig.CR_TO_WR_REWARD  #correct to wrong
STUBBORN_REWARD = routeConfig.STUBBORN_REWARD #stick to a certain correct channel

OBSERV_BATCH = routeConfig.OBSERV_BATCH

PATH_A_CNT = routeConfig.PATH_A_CNT
PATH_B_CNT = routeConfig.PATH_B_CNT
PATH_C_CNT = routeConfig.PATH_C_CNT
TOTAL_PATH_CNT = routeConfig.TOTAL_PATH_CNT

STATE_A_CNT = routeConfig.STATE_A_CNT #double of channal count
STATE_B_CNT = routeConfig.STATE_B_CNT
STATE_C_CNT = routeConfig.STATE_C_CNT

#BLOCK_CNT = channelConfig.BLOCK_CNT
USR_CNT = routeConfig.USR_CNT
REFRESH = routeConfig.REFRESH
#REFRESH_METHOD_OLD = channelConfig.REFRESH_METHOD_OLD
#JAMMER_TYPE = channelConfig.JAMMER_TYPE


class Jammer:
    def __init__(self):
        self.type = routeConfig.JAMMER_TYPE
        self.markov_config = routeConfig.MARKOV_CONFIG
        self.t_key = [[1,2,0],[2,0,1],[1,0,2]]

        self.block_cnt = routeConfig.BLOCK_CNT
        self.state = routeConfig.MARKOV_CONFIG

        self.channel_p = routeConfig.CHANNEL_P[self.state]
        #self.channel_p1 = routeConfig.CHANNEL_P1
        self.p_aggre=[0 for x in range(TOTAL_PATH_CNT+1)]
        for i in range (1,TOTAL_PATH_CNT+1):
            self.p_aggre[i] = self.channel_p[i-1] + self.p_aggre[i-1]

    def changeState(self):
        if self.type == 'Markov_jammer':
            ran = random.random()
            if self.state == 0:
                if ran > 0.9:
                    self.state = self.t_key[self.markov_config][0]
                elif ran > 0.8:
                    self.state = self.t_key[self.markov_config][1]
                else:
                    self.state = self.t_key[self.markov_config][2]
            elif self.state == 1:
                if ran > 0.9:
                    self.state = self.t_key[self.markov_config][0]
                elif ran > 0.8:
                    self.state = self.t_key[self.markov_config][1]
                else:
                    self.state = self.t_key[self.markov_config][2]
            else:
                if ran > 0.9:
                    self.state = self.t_key[self.markov_config][0]
                elif ran > 0.8:
                    self.state = self.t_key[self.markov_config][1]
                else:
                    self.state = self.t_key[self.markov_config][2]
            self.channel_p = routeConfig.CHANNEL_P[self.state]
            print (self.channel_p)
            for i in range (1,TOTAL_PATH_CNT+1):
                self.p_aggre[i] = self.channel_p[i-1] + self.p_aggre[i-1]


    def act(self):
        #self.changeState()
        channel_available = dict()
        for i in range(TOTAL_PATH_CNT):
            channel_available[i+1] = 1
        if self.type == 'Random_jammer_1':
            # A jammer blockes certain channels with fixed probability
            # The number of jammed channels is different at each timeslot
            for i in range (TOTAL_PATH_CNT):
                ran = random.random()
                #print("path:", i+1)
                if ran < self.channel_p[i]:
                    #print("random num:", ran)
                    #print("channel_p:", self.channel_p1[i])
                    #print("channel blocked")
                    channel_available[i+1] = 0         #this channel blocked
                else:
                    #print("random num:", ran)
                    #print("channel_p:", self.channel_p1[i])
                    #print("channel available")
                    channel_available[i+1] = 1         #this channel available
        elif self.type == 'Markov_jammer':
            # A jammer blockes 3 channels at each timeslot with fixed probability
            #ran = random.random()
           # for i in range (TOTAL_PATH_CNT):
           #     if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
           #         channel_available[i+1] = 0
           # available_cnt = TOTAL_PATH_CNT - 1
           # while available_cnt > (TOTAL_PATH_CNT-self.block_cnt):
           #     ran = random.random()
           #     for i in range (TOTAL_PATH_CNT):
           #         if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
           #             channel_available[i+1] = 0
           #     available_cnt = 0
           #     for i in range (TOTAL_PATH_CNT):
           #         available_cnt += channel_available[i+1]
            self.changeState()
            for i in range (TOTAL_PATH_CNT):
                ran = random.random()
                if ran < self.channel_p[i]:
                    channel_available[i+1] = 0         #this channel blocked
                else:
                    channel_available[i+1] = 1         #this channel available

        else:
            print ("No jammer named:" + JAMMER_TYPE)
        return channel_available


class RouteEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 2
    }

    def __init__(self):
        self.states_a = range(1,STATE_A_CNT+1) #状态空间
        self.states_b = range(1,STATE_B_CNT+1) #for a, total 9 states, 123 blocked 456 available 789 occupied
        self.states_c = range(1,STATE_C_CNT+1)

        self.xa=[0 for x in range(PATH_A_CNT)]
        self.ya=[0 for x in range(PATH_A_CNT)]
        self.xb=[0 for x in range(PATH_B_CNT)]
        self.yb=[0 for x in range(PATH_B_CNT)]
        self.xc=[0 for x in range(PATH_C_CNT)]
        self.yc=[0 for x in range(PATH_C_CNT)]

        self.path_a_cnt = PATH_A_CNT
        self.path_b_cnt = PATH_B_CNT
        self.path_c_cnt = PATH_C_CNT

        self.usr_cnt = USR_CNT
        self.total_path_cnt = TOTAL_PATH_CNT

        self.jammer = Jammer()
        #self.actionTranslationt = dict()
        self.actionTranslationt = self.actionTranslation() #################################################

        self.state_batch = [0 for x in range(OBSERV_BATCH)]

      #  for i in range (CHANNEL_CNT):   #画图位置参数
      #      if (i % 5 == 0):
      #          self.x[i] = 200
      #      elif (i % 5 == 1) :
      #          self.x[i] = 400
      #      elif (i % 5 == 2) :
      #          self.x[i] = 600
      #      elif (i % 5 == 3) :
      #          self.x[i] = 800
      #      else:
      #          self.x[i] = 1000

      #  for i in range (CHANNEL_CNT):
      #      self.y[i] = 1100 - 200 * int(i / 5)
        self.y=[900,800,700,600,500,400,300]
        #self.line_x_start=[900,800,700,600,500,400,300]
        #self.line_x_end=[900,800,700,600,500,400,300]
        self.line_y_start=[750,750,750,550,550,550,550,350,350,350,350]
        self.line_y_end=[900,800,700,800,700,600,500,600,500,400,300]

        self.jammer_y=[700,600,500]

        

        self.channel_p1 = self.jammer.channel_p#[0.2, 0.5, 0.8, 0.5, 0.2, 0.05, 0.02]#, 0.60, 0.98, 0.99, 0.90, 0.05, 0.80, 0.05, 0.91, 0.93, 0.80, 0.95, 0.92, 0.08, 0.90, 0.93, 0.99, 0.37, 0.95, 0.91, 0.99, 0.87, 0.04, 0.91] #jamming probability

        #终止状态为字典格式 #需初始化及动态更新

        self.channel_available = self.jammer.act()  #chnl_available: 0 for blocked, 1 for available, 2 for occupied
                                                    #jammer.act() only determines 0/1

        #self.actions = [0 for x in range(CHANNEL_CNT)]
        #for i in range (CHANNEL_CNT):
        #    self.actions[i] = i+1

        self.rewards = dict()        #回报的数据结构为字典
        #self.calculateReward()

        self.t_a = dict()             #状态转移的数据格式为字典
        self.t_b = dict()
        self.t_c = dict()
        self.updateStateTransfert()

        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None


    #def getTerminal(self):
    #    return self.channel_available

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    #def getAction(self):
    #    return self.actions
    def getChannelAvailable(self):
        return self.channel_available
    def setAction(self,s):
        self.state=s
    def getRewardChart(self):
        print ("\nReward Chart\n", self.rewards)
    def getTChart(self):
        print ("\nTransformation Chart\n", self.t)

    def isChannelBlocked(self, s, who): 
        if (who == 'a'):# get path availability for abc three users
            if (s <= self.path_a_cnt):   #for a, total 9 states, 123 blocked 456 available 789 occupied
                return True
            else:
                return False
        elif (who == 'b'):# get channel availability
            if (s <= self.path_b_cnt):
                return True
            else:
                return False
        elif (who == 'c'):# get channel availability
            if (s <= self.path_c_cnt):
                return True
            else:
                return False
        else:
            print ("Syntax Error. cannot check availability!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def isChannelOccupied(self, s, who): #input is state
        if (who == 'a'):# get path availability for abc three users
            if (s > self.path_a_cnt*2):
                return True
            else:
                return False
        elif (who == 'b'):# get channel availability
            if (s > self.path_b_cnt*2):
                return True
            else:
                return False
        elif (who == 'c'):# get channel availability
            if (s > self.path_c_cnt*2):
                return True
            else:
                return False
        else:
            print ("Syntax Error. cannot check occupation status!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def getChannelNumber(self, s, who):
        # get channel number from state
        if (who == 'a'):# get path availability for abc three users
            if (s % PATH_A_CNT == 0):
                return s % PATH_A_CNT + 3
            else:
                return s % PATH_A_CNT
        elif (who == 'b'):# get channel availability
            if (s % PATH_B_CNT == 0):
                return s % PATH_B_CNT + 4 + 1
            else:
                return s % PATH_B_CNT + 1
        elif (who == 'c'):# get channel availability
            if (s % PATH_C_CNT == 0):
                return s % PATH_C_CNT + 4 + 3
            else:
                return s % PATH_C_CNT + 3
        else:
            print ("Syntax Error. cannot get path number!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def setStateBatch(self, s):
        self.state_batch = s

    #def calculateReward(self): ##################################################################
    #    for i in range (STATE_CNT):   #i in 0-9 means blocked, 10-19 means available
    #        for j in range (CHANNEL_CNT):
    #            key = "%d-%d"%(i+1, j+1)
    #            if self.isChannelBlocked(i):             #i is blocked
    #                if (self.channel_available[j+1] == 1):
    #                    self.rewards[key] = CORRECT_REWARD       #wrong to correct
    #                elif (self.channel_available[j+1] == 0):
    #                    if (i==j):
    #                        self.rewards[key] = PUNISH_REWARD     #wrong to itself
    #                    else:
    #                        self.rewards[key] = WR_TO_WR_REWARD   #wrong to another wrong
    #            else:                             # i-10 is available
    #                if (self.channel_available[j+1] == 0):
    #                    self.rewards[key] = CR_TO_WR_REWARD           #correct to wrong
    #                elif (self.channel_available[j+1] == 1):
    #                    if (self.getChannelNumber(i) == j):
    #                        self.rewards[key] = CORRECT_REWARD    #correct to itself
    #                    else:
    #                        self.rewards[key] = CR_TO_CR_REWARD   #correct to another correct

    def updateStateTransfert(self):   
        for i in range (STATE_A_CNT):
            for j in range (PATH_A_CNT): #i+1 is current state, j+1 is action, self.t_a[key] is next_state
                key = "%d-%d"%(i+1, j+1)
                if (self.channel_available[j+1] == 0):   #chnl_available: 0 for blocked, 1 for available, 2 for occupied
                    self.t_a[key] = j + 1
                elif (self.channel_available[j+1] == 1):
                    self.t_a[key] = j + 1 + PATH_A_CNT
                elif (self.channel_available[j+1] == 2):
                    self.t_a[key] = j + 1 + PATH_A_CNT + PATH_A_CNT
        for i in range (STATE_B_CNT):
            for j in range (PATH_B_CNT): 
                key = "%d-%d"%(i+1, j+1)
                if (self.channel_available[j+2] == 0): #j+2 is path number
                    self.t_b[key] = j + 1
                elif (self.channel_available[j+2] == 1):
                    self.t_b[key] = j + 1 + PATH_B_CNT
                elif (self.channel_available[j+2] == 2):
                    self.t_b[key] = j + 1 + PATH_B_CNT + PATH_B_CNT
        for i in range (STATE_C_CNT):
            for j in range (PATH_C_CNT): 
                key = "%d-%d"%(i+1, j+1)
                if (self.channel_available[j+4] == 0): #j+4 is path number
                    self.t_c[key] = j + 1
                elif (self.channel_available[j+4] == 1):
                    self.t_c[key] = j + 1 + PATH_C_CNT
                elif (self.channel_available[j+4] == 2):
                    self.t_c[key] = j + 1 + PATH_C_CNT + PATH_C_CNT

    def pathOccupation(self, state, who):
        num = self.getChannelNumber(state, who)
        self.channel_available[num] = 2

    def actionTranslation(self):
        actionTranslationTable = dict()
        action_bar = [1, 1, 0]
        for i in range (self.path_a_cnt * self.path_b_cnt * self.path_c_cnt): #0-47
            #print(i+1)
            action_bar[2] += 1
            if (action_bar[2] > self.path_c_cnt):
                action_bar[2] = 1
                action_bar[1] += 1
            if (action_bar[1] > self.path_b_cnt):
                action_bar[1] = 1
                action_bar[0] += 1
            actionTranslationTable[i+1] = [action_bar[0], action_bar[1], action_bar[2]]
        #print(actionTranslationTable)

        return actionTranslationTable

    def step(self, action):
            #系统当前状态
        state_a = self.state_a
        state_b = self.state_b
        state_c = self.state_c
        print("present state:", state_a, state_b, state_c)
        #print("present state:", self.state_batch[0])

        [action_a, action_b, action_c] = self.actionTranslationt[action]
        print ("action of abc:",[action_a, action_b, action_c])

        key_a = "%d-%d"%(state_a, action_a)   #将状态和动作组成字典的键值
        key_b = "%d-%d"%(state_b, action_b)
        key_c = "%d-%d"%(state_c, action_c)

        print("key_a:", key_a)
        print("key_b:", key_b)
        print("key_c:", key_c)
        #print("reward:",self.rewards[key])
            #状态转移
        if (key_a in self.t_a):
            print("key_a in dict:", key_a)
            next_state_a = self.t_a[key_a]
            print("next state_a:", next_state_a)
            self.pathOccupation(next_state_a, 'a')
            print("path occupied:", self.getChannelNumber(next_state_a, 'a'))
        else:###############
            print("key not in dict:", key_a)
            next_state_a = state_a
        self.updateStateTransfert()

        if (key_b in self.t_b):
            print("key_a in dict:", key_b)
            next_state_b = self.t_b[key_b]
            print("next state_b:", next_state_b)
            self.pathOccupation(next_state_b, 'b')
            print("path occupied:", self.getChannelNumber(next_state_b, 'b'))
        else:
            print("key not in dict:", key_b)
            next_state_b = state_b
        self.updateStateTransfert()

        if (key_c in self.t_c):
            print("key_a in dict:", key_c)
            next_state_c = self.t_c[key_c]
            print("next state_c:", next_state_c)
        else:
            print("key not in dict:", key_c)
            next_state_c = state_c

        self.state_a = next_state_a
        self.state_b = next_state_b
        self.state_c = next_state_c

        is_terminal = False

#        if self.channel_available[next_state]:
        #if (next_state > CHANNEL_CNT): ######################################################从这开始 这个值目前没作用
        #           is_terminal = True #was True

            #reward 定义
        #if key not in self.rewards:
        #    print ("key not in reward dict")
        #    r = 0.0
        #else:
        #    print ("key in reward dict")
        #    r = self.rewards[key]

   ###########in case of markov ################################################ set reward
        r = 0
        #if (self.jammer.type == 'Random_jammer_1'):
            #avail_sum_state = OBSERV_BATCH
            #avail_sum_state_next = OBSERV_BATCH
                                                             
        if (self.isChannelBlocked(self.state_a, 'a')):     
            r -= 1
        elif (self.isChannelOccupied(self.state_a, 'a')):     
            r += 0.5
        else:
            r += 2.5

        if (self.isChannelBlocked(self.state_b, 'b')):     
            r -= 1
        elif (self.isChannelOccupied(self.state_b, 'b')):     
            r += 0.5
        else:
            r += 2

        if (self.isChannelBlocked(self.state_c, 'c')):     
            r -= 1
        elif (self.isChannelOccupied(self.state_c, 'c')):     
            r += 0.5
        else:
            r += 2.5

            #for i in range (OBSERV_BATCH-1):
            #    if (self.isChannelBlocked(self.state_batch[i])):     
            #        avail_sum_state_next -= 1                        
            #if (self.isChannelBlocked(next_state)):                  
            #    avail_sum_state_next -= 1
            #print(avail_sum_state)
            #print(avail_sum_state_next)

            #if (avail_sum_state_next == OBSERV_BATCH):
           #     r = 0.15
           # elif (avail_sum_state_next == OBSERV_BATCH-1):
           #     r = 0.1
          #  elif ((avail_sum_state_next == OBSERV_BATCH-1) and (avail_sum_state == OBSERV_BATCH)):
         #       r = -100#CORRECT_REWARD
          #  else:
          #      r = -2

            #r = avail_sum_state_next *2 -1.9
        #############################################################################################################
        #刷新信道使用状态
        self.refresh()

        return [np.array(next_state_a), np.array(next_state_b), np.array(next_state_c)], r, 0 ,{}

    def refresh(self):
    #刷新信道使用状态
        if (REFRESH):
            self.channel_available = self.jammer.act()
    #reward update
            #self.calculateReward()
    #t update
            self.updateStateTransfert()

    def reset(self):
        i = int(random.random() * PATH_A_CNT) + 1
        if self.channel_available[i]:
            self.state_a = i + PATH_A_CNT
        else:
            self.state_a = i

        i = int(random.random() * PATH_B_CNT) + 1
        if self.channel_available[i+1]:
            self.state_b = i + PATH_B_CNT
        else:
            self.state_b = i

        i = int(random.random() * PATH_C_CNT) + 1
        if self.channel_available[i+3]:
            self.state_c = i + PATH_C_CNT
        else:
            self.state_c = i

        return [np.array(self.state_a), np.array(self.state_b), np.array(self.state_c)]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 1200
        screen_height = 1200

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建网格世界
            #self.line1 = rendering.Line((100,300),(500,300))
                #创建信道模型1
            self.line = []
            for i in range(PATH_A_CNT+PATH_B_CNT+PATH_C_CNT):
                self.line.append(rendering.Line((200,self.line_y_start[i]),(400,self.line_y_end[i])))
                self.line.append(rendering.Line((1000,self.line_y_start[i]),(800,self.line_y_end[i])))

            self.s1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(200,750))
            self.s1.add_attr(self.circletrans)
            self.s1.set_color(0,255,255)

            self.s2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(200,550))
            self.s2.add_attr(self.circletrans)
            self.s2.set_color(0,0,255)

            self.s3 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(200,350))
            self.s3.add_attr(self.circletrans)
            self.s3.set_color(255,0,255)

            self.d1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(1000,750))
            self.d1.add_attr(self.circletrans)
            self.d1.set_color(0,255,255)

            self.d2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(1000,550))
            self.d2.add_attr(self.circletrans)
            self.d2.set_color(0,0,255)

            self.d3 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(1000,350))
            self.d3.add_attr(self.circletrans)
            self.d3.set_color(255,0,255)

            self.path = []
            for i in range(TOTAL_PATH_CNT):
                self.path.append(rendering.Line((400,self.y[i]),(800,self.y[i])))
                #self.pathtrans = rendering.LineWidth(10)
                #self.circletrans = rendering.Transform(translation=(self.x[i], self.y[i]))
                #self.path[i].linewidth = rendering.LineWidth(stroke = 10)#(self.pathtrans)
                #self.path[i].add_attr(self.path[i].linewidth)
                if self.channel_available[i + 1] == 1:
                    self.path[i].set_color(0,255,0)        #available - green
                elif self.channel_available[i + 1] == 2:
                    self.path[i].set_color(255,255,0)        #occupied - yellow
                else:
                    self.path[i].set_color(255,0,0)        #blocked - red

                    #创建机器人
            self.robot_a= rendering.make_circle(30)
            self.robotrans_a = rendering.Transform()
            self.robot_a.add_attr(self.robotrans_a)
            self.robot_a.set_color(0, 255, 255)  #cyan

            self.robot_b= rendering.make_circle(30)
            self.robotrans_b = rendering.Transform()
            self.robot_b.add_attr(self.robotrans_b)
            self.robot_b.set_color(0, 0, 255)  #blue

            self.robot_c= rendering.make_circle(30)
            self.robotrans_c = rendering.Transform()
            self.robot_c.add_attr(self.robotrans_c)
            self.robot_c.set_color(255, 0, 255)  #magenta

            self.robot_jammer= rendering.make_capsule(0,450)
            self.robotrans_jammer = rendering.Transform()
            self.robot_jammer.add_attr(self.robotrans_jammer)
            self.robot_jammer.set_color(255,255,0)  #

            self.viewer.add_geom(self.robot_jammer)

            for i in range(TOTAL_PATH_CNT):
                self.viewer.add_geom(self.path[i])

            self.viewer.add_geom(self.robot_a)
            self.viewer.add_geom(self.robot_b)
            self.viewer.add_geom(self.robot_c)

            for i in range (2*(PATH_A_CNT+PATH_B_CNT+PATH_C_CNT)):
                self.viewer.add_geom(self.line[i])

            self.viewer.add_geom(self.s1)
            self.viewer.add_geom(self.s2)
            self.viewer.add_geom(self.s3)
            self.viewer.add_geom(self.d1)
            self.viewer.add_geom(self.d2)
            self.viewer.add_geom(self.d3)

        else:
            for i in range(TOTAL_PATH_CNT):
                if self.channel_available[i + 1] == 1:
                    self.path[i].set_color(0,255,0)        #available - green
                elif self.channel_available[i + 1] == 2:
                    self.path[i].set_color(255,255,0)        #occupied - yellow
                else:
                    self.path[i].set_color(255,0,0)


        if self.state_a is None: return None
        if self.state_b is None: return None
        if self.state_c is None: return None

        #self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        
        if (self.state_a > PATH_A_CNT*2):
            self.robotrans_a.set_translation(500, self.y[self.state_a-1-PATH_A_CNT-PATH_A_CNT])
        elif(self.state_a <= PATH_A_CNT):
            self.robotrans_a.set_translation(500, self.y[self.state_a-1])
        else:
            self.robotrans_a.set_translation(500, self.y[self.state_a-1-PATH_A_CNT])
        
        if (self.state_b > PATH_B_CNT*2):
            self.robotrans_b.set_translation(600, self.y[self.state_b-1-PATH_B_CNT-PATH_B_CNT+1])
        elif(self.state_b <= PATH_B_CNT):
            self.robotrans_b.set_translation(600, self.y[self.state_b-1+1])
        else:
            self.robotrans_b.set_translation(600, self.y[self.state_b-1-PATH_B_CNT+1])
        
        if (self.state_c > PATH_C_CNT*2):
            self.robotrans_c.set_translation(700, self.y[self.state_c-1-PATH_C_CNT-PATH_C_CNT+3])
        elif(self.state_c <= PATH_C_CNT):
            self.robotrans_c.set_translation(700, self.y[self.state_c-1+3])
        else:
            self.robotrans_c.set_translation(700, self.y[self.state_c-1-PATH_C_CNT+3])

        self.robotrans_jammer.set_translation(600, self.jammer_y[self.jammer.state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
