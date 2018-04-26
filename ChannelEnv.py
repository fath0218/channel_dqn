import math
import gym
import logging
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

CORRECT_REWARD = 10.0
PUNISH_REWARD = -50.0
C_TO_C_REWARD = -60.0

CHANNEL_CNT = 10
BLOCK_CNT = 3
USR_CNT = 3
REFRESH = 1

class ChannelEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 2
	}
	
	def __init__(self):
		self.states = range(1,CHANNEL_CNT+1) #状态空间
		self.available_cnt = CHANNEL_CNT
		self.x=[0 for x in range(CHANNEL_CNT)]
		self.y=[0 for x in range(CHANNEL_CNT)]
		self.channel_cnt = CHANNEL_CNT
		for i in range (CHANNEL_CNT):
			if (i % 5 == 0):
				self.x[i] = 200        
			elif (i % 5 == 1) :
				self.x[i] = 400
			elif (i % 5 == 2) :
				self.x[i] = 600
			elif (i % 5 == 3) :
				self.x[i] = 800
			else:
				self.x[i] = 1000
		
		for i in range (CHANNEL_CNT):
			self.y[i] = 1100 - 200 * int(i / 5)
        
##########################################################################################################
#		self.channel_p = [0.9, 0.11, 0.85, 0.92, 0.95, 0.99, 0.8, 0.60, 0.98, 0.99, 0.90, 0.05, 0.80, 0.05, 0.91, 0.93, 0.80, 0.95, 0.92, 0.08, 0.90, 0.93, 0.99, 0.37, 0.95, 0.91, 0.99, 0.87, 0.04, 0.91] #jamming probability
		
		#终止状态为字典格式 #需初始化及动态更新
		self.terminate_states = dict()
		for i in range(CHANNEL_CNT):
			self.terminate_states[i+1] = 1
#		for i in range (CHANNEL_CNT):
#			ran = random.random()
#			#print("channel:", i+1)
#			if ran < self.channel_p[i]:
				#print("random num:", ran)
				#print("channel_p:", self.channel_p[i])
				#print("channel blocked")
#				self.terminate_states[i+1] = 0         #this channel blocked
#			else:
				#print("random num:", ran)
				#print("channel_p:", self.channel_p[i])
				#print("channel available")
#				self.terminate_states[i+1] = 1         #this channel available
##########################################################################################################
		self.channel_p = [0.22, 0.21, 0.05, 0.02, 0.25, 0.14, 0.02, 0.01, 0.04, 0.04]
		self.p_aggre=[0 for x in range(CHANNEL_CNT+1)]
		
		for i in range (1,CHANNEL_CNT+1):
			self.p_aggre[i] = self.channel_p[i-1] + self.p_aggre[i-1]

		while self.available_cnt > (CHANNEL_CNT-BLOCK_CNT):
			ran = random.random()
			for i in range (CHANNEL_CNT):
				if (self.terminate_states[i+1] == 1):
					if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
						self.terminate_states[i+1] = 0
					else:
						self.terminate_states[i+1] = 1
			self.available_cnt = 0
			for i in range (CHANNEL_CNT):
				self.available_cnt += self.terminate_states[i+1]
##########################################################################################################
		
		self.actions = [0 for x in range(CHANNEL_CNT)]
		for i in range (CHANNEL_CNT):
			self.actions[i] = i+1

		self.rewards = dict();        #回报的数据结构为字典
		for i in range (CHANNEL_CNT):
			for j in range (CHANNEL_CNT):
				key = "%d-%d"%(i+1, j+1)
				if (self.terminate_states[i+1] == 0) and (self.terminate_states[j+1] == 1):
					self.rewards[key] = CORRECT_REWARD        #correct to correct
				elif (self.terminate_states[i+1] == 0) and (i==j) :
					self.rewards[key] = PUNISH_REWARD       #wrong to itself
				else: #elif (self.terminate_states[i+1] == 0) and (self.terminate_states[j+1] == 0):
					self.rewards[key] = 0.0
		
		self.t = dict();             #状态转移的数据格式为字典
		for i in range (CHANNEL_CNT):
			for j in range (CHANNEL_CNT):
				key = "%d-%d"%(i+1, j+1)
				self.t[key] = j+1        

		self.gamma = 0.8         #折扣因子
		self.viewer = None
		self.state = None
		
		
	def getTerminal(self):
		return self.terminate_states

	def getGamma(self):
		return self.gamma
	
	def getStates(self):
		return self.states
	
	def getAction(self):
		return self.actions
	def getTerminate_states(self):
		return self.terminate_states
	def setAction(self,s):
		self.state=s
		
	def step(self, action):
        	#系统当前状态
		state = self.state
		print("present state:", state)
		if self.terminate_states[state]:
			print("this is the terminal state")
			return state, 1.0, True, {}
		key = "%d-%d"%(state, action)   #将状态和动作组成字典的键值
		print("key:", key)
	    	#状态转移
		if key in self.t:
			print("key in dict:", key)
			next_state = self.t[key]
			print("next state:", next_state)
		else:
			print("key not in dict:", key)
			next_state = state
		self.state = next_state
		
		is_terminal = False
		
		if self.terminate_states[next_state]:
           		is_terminal = True
			
		if key not in self.rewards:
			r = 0.0
		else:
			r = self.rewards[key]

		#刷新信道使用状态
		if (REFRESH):	
###############################################################################################
#			for i in range (CHANNEL_CNT):
#				ran = random.random()
#				#print("channel:", i+1)
#				if ran < self.channel_p[i]:
					#print("random num:", ran)
					#print("channel_p:", self.channel_p[i])
					#print("channel blocked")
#					self.terminate_states[i+1] = 0         #this channel blocked
#				else:
					#print("random num:", ran)
					#print("channel_p:", self.channel_p[i])
					#print("channel available")
#					self.terminate_states[i+1] = 1         #this channel available
##############################################################################################
			print (self.available_cnt)
			ran = random.random()
			for i in range (CHANNEL_CNT):
				if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
					self.terminate_states[i+1] = 0
				else:
					self.terminate_states[i+1] = 1
			self.available_cnt = CHANNEL_CNT - 1
			while self.available_cnt > (CHANNEL_CNT-BLOCK_CNT):
				ran = random.random()
				for i in range (CHANNEL_CNT):
					if (self.terminate_states[i+1] == 1):
						if (ran >= self.p_aggre[i]) and (ran < self.p_aggre[i+1]):
							self.terminate_states[i+1] = 0
						else:
							self.terminate_states[i+1] = 1
				self.available_cnt = 0
				for i in range (CHANNEL_CNT):
					self.available_cnt += self.terminate_states[i+1]
##############################################################################################
			for i in range (CHANNEL_CNT):
				for j in range (CHANNEL_CNT):
					key = "%d-%d"%(i+1, j+1)
					if (self.terminate_states[i+1] == 0) and (self.terminate_states[j+1] == 1):
						self.rewards[key] = CORRECT_REWARD        #available - green
					elif (self.terminate_states[i+1] == 0) and (i==j) :
						self.rewards[key] = PUNISH_REWARD
					else:
						self.rewards[key] = 0.0


		return np.array(next_state), r,is_terminal,{}
		
	def reset(self):
		self.state = self.states[int(random.random() * len(self.states))]
		return np.array(self.state)
		
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
			self.chnl = []
			for i in range(CHANNEL_CNT):
				self.chnl.append(rendering.make_circle(50))
				self.circletrans = rendering.Transform(translation=(self.x[i], self.y[i]))
				self.chnl[i].add_attr(self.circletrans)
				if self.terminate_states[i + 1]:
					self.chnl[i].set_color(0,255,0)        #available - green
				else:
					self.chnl[i].set_color(255,0,0) #blocked - red	
            		#创建机器人
			self.robot= rendering.make_circle(30)
			self.robotrans = rendering.Transform()
			self.robot.add_attr(self.robotrans)
			self.robot.set_color(255, 255, 255)
			
			for i in range(CHANNEL_CNT):
				self.viewer.add_geom(self.chnl[i])

			self.viewer.add_geom(self.robot)

		else:
			for i in range(CHANNEL_CNT):
				if self.terminate_states[i + 1]:
					self.chnl[i].set_color(0,255,0)        #available - green
				else:
					self.chnl[i].set_color(255,0,0) #blocked - red
			

		if self.state is None: return None
		#self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
		self.robotrans.set_translation(self.x[self.state-1], self.y[self.state-1])
		
		
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')
