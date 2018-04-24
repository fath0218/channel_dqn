import math
import gym
import logging
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

RIGHT_REWARD = 10.0
PUNISH_REWARD = -500.0
CHANNEL_CNT = 30
REFRESH = 1
class ChannelEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 2
	}
	
	def __init__(self):
		self.states = range(1,CHANNEL_CNT+1) #状态空间
		self.x=[0 for x in range(CHANNEL_CNT)]
		self.y=[0 for x in range(CHANNEL_CNT)]
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

		self.channel_p = [0.9, 0.11, 0.85, 0.92, 0.95, 0.99, 0.8, 0.60, 0.98, 0.99, 0.90, 0.05, 0.80, 0.05, 0.91, 0.93, 0.80, 0.95, 0.92, 0.08, 0.90, 0.93, 0.99, 0.37, 0.95, 0.91, 0.99, 0.87, 0.04, 0.91] #zuse gailv
		
		#终止状态为字典格式 #需初始化及动态更新
		self.terminate_states = dict()
		for i in range (CHANNEL_CNT):
			ran = random.random()
			#print("channel:", i+1)
			if ran < self.channel_p[i]:
				#print("random num:", ran)
				#print("channel_p:", self.channel_p[i])
				#print("channel blocked")
				self.terminate_states[i+1] = 0         #this channel blocked
			else:
				#print("random num:", ran)
				#print("channel_p:", self.channel_p[i])
				#print("channel available")
				self.terminate_states[i+1] = 1         #this channel available

		
		self.actions = [0 for x in range(CHANNEL_CNT)]
		for i in range (CHANNEL_CNT):
			self.actions[i] = i+1

		self.rewards = dict();        #回报的数据结构为字典
		for i in range (CHANNEL_CNT):
			for j in range (CHANNEL_CNT):
				key = "%d-%d"%(i+1, j+1)
				if (self.terminate_states[i+1] == 0) and (self.terminate_states[j+1] == 1):
					self.rewards[key] = RIGHT_REWARD        #available - green
				elif (self.terminate_states[i+1] == 0) and (i==j) :
					self.rewards[key] = PUNISH_REWARD
				else:
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
			for i in range (CHANNEL_CNT):
				ran = random.random()
				#print("channel:", i+1)
				if ran < self.channel_p[i]:
					#print("random num:", ran)
					#print("channel_p:", self.channel_p[i])
					#print("channel blocked")
					self.terminate_states[i+1] = 0         #this channel blocked
				else:
					#print("random num:", ran)
					#print("channel_p:", self.channel_p[i])
					#print("channel available")
					self.terminate_states[i+1] = 1         #this channel available
			for i in range (CHANNEL_CNT):
				for j in range (CHANNEL_CNT):
					key = "%d-%d"%(i+1, j+1)
					if (self.terminate_states[i+1] == 0) and (self.terminate_states[j+1] == 1):
						self.rewards[key] = RIGHT_REWARD        #available - green
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
					self.chnl[i].set_color(255,0,0)        #blocked - red	

#            		#创建第二个信道
#			self.chnl2 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[1], self.y[1]))
#			self.chnl2.add_attr(self.circletrans)
#			if self.terminate_states[2]:
#				self.chnl2.set_color(0,255,0)        #available - green
#			else:
#				self.chnl2.set_color(255,0,0)        #blocked - red
#			
#			self.chnl3 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[2], self.y[2]))
#			self.chnl3.add_attr(self.circletrans)
#			if self.terminate_states[3]:
#				self.chnl3.set_color(0,255,0)        #available - green
#			else:
#				self.chnl3.set_color(255,0,0)        #blocked - red
#			
#			self.chnl4 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[3], self.y[3]))
#			self.chnl4.add_attr(self.circletrans)
#			if self.terminate_states[4]:
#				self.chnl4.set_color(0,255,0)        #available - green
#			else:
#				self.chnl4.set_color(255,0,0)        #blocked - red
#			
#			self.chnl5 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[4], self.y[4]))
#			self.chnl5.add_attr(self.circletrans)
#			if self.terminate_states[5]:
#				self.chnl5.set_color(0,255,0)        #available - green
#			else:
#				self.chnl5.set_color(255,0,0)        #blocked - red
#			
#			self.chnl6 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[5], self.y[5]))
#			self.chnl6.add_attr(self.circletrans)
#			if self.terminate_states[6]:
#				self.chnl6.set_color(0,255,0)        #available - green
#			else:
#				self.chnl6.set_color(255,0,0)        #blocked - red
#			
#			self.chnl7 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[6], self.y[6]))
#			self.chnl7.add_attr(self.circletrans)
#			if self.terminate_states[7]:
#				self.chnl7.set_color(0,255,0)        #available - green
#			else:
#				self.chnl7.set_color(255,0,0)        #blocked - red
#			
#			self.chnl8 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[7], self.y[7]))
#			self.chnl8.add_attr(self.circletrans)
#			if self.terminate_states[8]:
#				self.chnl8.set_color(0,255,0)        #available - green
#			else:
#				self.chnl8.set_color(255,0,0)        #blocked - red
#			
#			self.chnl9 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[8], self.y[8]))
#			self.chnl9.add_attr(self.circletrans)
#			if self.terminate_states[9]:
#				self.chnl9.set_color(0,255,0)        #available - green
#			else:
#				self.chnl9.set_color(255,0,0)        #blocked - red
#			
#			self.chnl10 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[9], self.y[9]))
#			self.chnl10.add_attr(self.circletrans)
#			if self.terminate_states[10]:
#				self.chnl10.set_color(0,255,0)        #available - green
#			else:
#				self.chnl10.set_color(255,0,0)        #blocked - red
#
#		##创建第11个信道
#			self.chnl11 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[10], self.y[10]))
#			self.chnl11.add_attr(self.circletrans)
#			if self.terminate_states[11]:
#				self.chnl11.set_color(0,255,0)        #available - green
#			else:
#				self.chnl11.set_color(255,0,0)        #blocked - red	
#
#			self.chnl12 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[11], self.y[11]))
#			self.chnl12.add_attr(self.circletrans)
#			if self.terminate_states[12]:
#				self.chnl12.set_color(0,255,0)        #available - green
#			else:
#				self.chnl12.set_color(255,0,0)        #blocked - red
#			
#			self.chnl13 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[12], self.y[12]))
#			self.chnl13.add_attr(self.circletrans)
#			if self.terminate_states[13]:
#				self.chnl13.set_color(0,255,0)        #available - green
#
#			else:
#				self.chnl13.set_color(255,0,0)        #blocked - red
#			
#			self.chnl14 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[13], self.y[13]))
#			self.chnl14.add_attr(self.circletrans)
#			if self.terminate_states[14]:
#				self.chnl14.set_color(0,255,0)        #available - green
#			else:
#				self.chnl14.set_color(255,0,0)        #blocked - red
#			
#			self.chnl15 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[14], self.y[14]))
#			self.chnl15.add_attr(self.circletrans)
#			if self.terminate_states[15]:
#				self.chnl15.set_color(0,255,0)        #available - green
#			else:
#				self.chnl15.set_color(255,0,0)        #blocked - red
#			
#			self.chnl16 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[15], self.y[15]))
#			self.chnl16.add_attr(self.circletrans)
#			if self.terminate_states[16]:
#				self.chnl16.set_color(0,255,0)        #available - green
#			else:
#				self.chnl16.set_color(255,0,0)        #blocked - red
#			
#			self.chnl17 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[16], self.y[16]))
#			self.chnl17.add_attr(self.circletrans)
#			if self.terminate_states[17]:
#				self.chnl17.set_color(0,255,0)        #available - green
#			else:
#				self.chnl17.set_color(255,0,0)        #blocked - red
#			
#			self.chnl18 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[17], self.y[17]))
#			self.chnl18.add_attr(self.circletrans)
#			if self.terminate_states[18]:
#				self.chnl18.set_color(0,255,0)        #available - green
#			else:
#				self.chnl18.set_color(255,0,0)        #blocked - red
#			
#			self.chnl19 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[18], self.y[18]))
#			self.chnl19.add_attr(self.circletrans)
#			if self.terminate_states[19]:
#				self.chnl19.set_color(0,255,0)        #available - green
#			else:
#				self.chnl19.set_color(255,0,0)        #blocked - red
#			
#			self.chnl20 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[19], self.y[19]))
#			self.chnl20.add_attr(self.circletrans)
#			if self.terminate_states[20]:
#				self.chnl20.set_color(0,255,0)        #available - green
#			else:
#				self.chnl20.set_color(255,0,0)        #blocked - red
#		##创建第21个信道
#			self.chnl21 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[20], self.y[20]))
#			self.chnl21.add_attr(self.circletrans)
#			if self.terminate_states[21]:
#				self.chnl21.set_color(0,255,0)        #available - green
#			else:
#				self.chnl21.set_color(255,0,0)        #blocked - red	
#
#			self.chnl22 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[21], self.y[21]))
#			self.chnl22.add_attr(self.circletrans)
#			if self.terminate_states[22]:
#				self.chnl22.set_color(0,255,0)        #available - green
#			else:
#				self.chnl22.set_color(255,0,0)        #blocked - red
#			
#			self.chnl23 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[22], self.y[22]))
#			self.chnl23.add_attr(self.circletrans)
#			if self.terminate_states[23]:
#				self.chnl23.set_color(0,255,0)        #available - green
#			else:
#				self.chnl23.set_color(255,0,0)        #blocked - red
#			
#			self.chnl24 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[23], self.y[23]))
#			self.chnl24.add_attr(self.circletrans)
#			if self.terminate_states[24]:
#				self.chnl24.set_color(0,255,0)        #available - green
#			else:
#				self.chnl24.set_color(255,0,0)        #blocked - red
#			
#			self.chnl25 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[24], self.y[24]))
#			self.chnl25.add_attr(self.circletrans)
#			if self.terminate_states[25]:
#				self.chnl25.set_color(0,255,0)        #available - green
#			else:
#				self.chnl25.set_color(255,0,0)        #blocked - red
#			
#			self.chnl26 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[25], self.y[25]))
#			self.chnl26.add_attr(self.circletrans)
#			if self.terminate_states[26]:
#				self.chnl26.set_color(0,255,0)        #available - green
#			else:
#				self.chnl26.set_color(255,0,0)        #blocked - red
#			
#			self.chnl27 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[26], self.y[26]))
#			self.chnl27.add_attr(self.circletrans)
#			if self.terminate_states[27]:
#				self.chnl27.set_color(0,255,0)        #available - green
#			else:
#				self.chnl27.set_color(255,0,0)        #blocked - red
#			
#			self.chnl28 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[27], self.y[27]))
#			self.chnl28.add_attr(self.circletrans)
#			if self.terminate_states[28]:
#				self.chnl28.set_color(0,255,0)        #available - green
#			else:
#				self.chnl28.set_color(255,0,0)        #blocked - red
#			
#			self.chnl29 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[28], self.y[28]))
#			self.chnl29.add_attr(self.circletrans)
#			if self.terminate_states[29]:
#				self.chnl29.set_color(0,255,0)        #available - green
#			else:
#				self.chnl29.set_color(255,0,0)        #blocked - red
#			
#			self.chnl30 = rendering.make_circle(50)
#			self.circletrans = rendering.Transform(translation=(self.x[29], self.y[29]))
#			self.chnl30.add_attr(self.circletrans)
#			if self.terminate_states[30]:
#				self.chnl30.set_color(0,255,0)        #available - green
#			else:
#				self.chnl30.set_color(255,0,0)        #blocked - red
			
            		#创建机器人
			self.robot= rendering.make_circle(30)
			self.robotrans = rendering.Transform()
			self.robot.add_attr(self.robotrans)
			self.robot.set_color(255, 255, 255)
			
			for i in range(CHANNEL_CNT):
				self.viewer.add_geom(self.chnl[i])
			#self.viewer.add_geom(self.chnl1)
			#self.viewer.add_geom(self.chnl2)
			#self.viewer.add_geom(self.chnl3)
			#self.viewer.add_geom(self.chnl4)
			#self.viewer.add_geom(self.chnl5)
			#self.viewer.add_geom(self.chnl6)
			#self.viewer.add_geom(self.chnl7)
			#self.viewer.add_geom(self.chnl8)
			#self.viewer.add_geom(self.chnl9)
			#self.viewer.add_geom(self.chnl10)

			#self.viewer.add_geom(self.chnl11)
			#self.viewer.add_geom(self.chnl12)
			#self.viewer.add_geom(self.chnl13)
			#self.viewer.add_geom(self.chnl14)
			#self.viewer.add_geom(self.chnl15)
			#self.viewer.add_geom(self.chnl16)
			#self.viewer.add_geom(self.chnl17)
			#self.viewer.add_geom(self.chnl18)
			#self.viewer.add_geom(self.chnl19)
			#self.viewer.add_geom(self.chnl20)

			#self.viewer.add_geom(self.chnl21)
			#self.viewer.add_geom(self.chnl22)
			#self.viewer.add_geom(self.chnl23)
			#self.viewer.add_geom(self.chnl24)
			#self.viewer.add_geom(self.chnl25)
			#self.viewer.add_geom(self.chnl26)
			#self.viewer.add_geom(self.chnl27)
			#self.viewer.add_geom(self.chnl28)
			#self.viewer.add_geom(self.chnl29)
			#self.viewer.add_geom(self.chnl30)

			self.viewer.add_geom(self.robot)

		else:
			for i in range(CHANNEL_CNT):
				if self.terminate_states[i + 1]:
					self.chnl[i].set_color(0,255,0)        #available - green
				else:
					self.chnl[i].set_color(255,0,0)        #blocked - red

			#if self.terminate_states[2]:
			#	self.chnl2.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl2.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[3]:
			#	self.chnl3.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl3.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[4]:
			#	self.chnl4.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl4.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[5]:
			#	self.chnl5.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl5.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[6]:
			#	self.chnl6.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl6.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[7]:
			#	self.chnl7.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl7.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[8]:
			#	self.chnl8.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl8.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[9]:
			#	self.chnl9.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl9.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[10]:
			#	self.chnl10.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl10.set_color(255,0,0)        #blocked - red
			#
		#11
			#if self.terminate_states[11]:
			#	self.chnl11.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl11.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[12]:
			#	self.chnl12.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl12.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[13]:
			#	self.chnl13.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl13.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[14]:
			#	self.chnl14.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl14.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[15]:
			#	self.chnl15.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl15.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[16]:
			#	self.chnl16.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl16.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[17]:
			#	self.chnl17.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl17.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[18]:
			#	self.chnl18.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl18.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[19]:
			#	self.chnl19.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl19.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[20]:
			#	self.chnl20.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl20.set_color(255,0,0)        #blocked - red
		#21
			#if self.terminate_states[21]:
			#	self.chnl21.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl21.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[22]:
			#	self.chnl22.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl22.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[23]:
			#	self.chnl23.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl23.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[24]:
			#	self.chnl24.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl24.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[25]:
			#	self.chnl25.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl25.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[26]:
			#	self.chnl26.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl26.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[27]:
			#	self.chnl27.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl27.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[28]:
			#	self.chnl28.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl28.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[29]:
			#	self.chnl29.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl29.set_color(255,0,0)        #blocked - red

			#if self.terminate_states[30]:
			#	self.chnl30.set_color(0,255,0)        #available - green
			#else:
			#	self.chnl30.set_color(255,0,0)        #blocked - red
			
			
		
		if self.state is None: return None
		#self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
		self.robotrans.set_translation(self.x[self.state-1], self.y[self.state-1])
		
		
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

		
        
		
		
	
		
