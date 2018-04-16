import math
import gym
import logging
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

RIGHT_REWARD = 10.0
PUNISH_REWARD = -50.0
CHANNEL_CNT = 10
REFRESH = 1
class ChannelEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 2
	}
	
	def __init__(self):
		self.states = range(1,CHANNEL_CNT+1) #状态空间
		self.x=[200,400,600,800,1000,200,400,600,800,1000]
		self.y=[400,400,400,400,400,200,200,200,200,200]
		

		self.channel_p = [0.9, 0.01, 0.85, 0.02, 0.95, 0.99, 0.8, 0.03, 0.98, 0.99] #zuse gailv
		
		#终止状态为字典格式 #需初始化及动态更新
		self.terminate_states = dict()
		for i in range (CHANNEL_CNT):
			ran = random.random()
			print("channel:", i+1)
			if ran < self.channel_p[i]:
				print("random num:", ran)
				print("channel_p:", self.channel_p[i])
				print("channel blocked")
				self.terminate_states[i+1] = 0         #this channel blocked
			else:
				print("random num:", ran)
				print("channel_p:", self.channel_p[i])
				print("channel available")
				self.terminate_states[i+1] = 1         #this channel available

		
		self.actions = [1,2,3,4,5,6,7,8,9,10]

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
		self.t['1-1'] = 1
		self.t['2-1'] = 1	
		self.t['3-1'] = 1	
		self.t['4-1'] = 1
		self.t['5-1'] = 1
		self.t['6-1'] = 1	
		self.t['7-1'] = 1
		self.t['8-1'] = 1	
		self.t['9-1'] = 1	
		self.t['10-1'] = 1	
		self.t['1-2'] = 2
		self.t['2-2'] = 2
		self.t['3-2'] = 2
		self.t['4-2'] = 2	
		self.t['5-2'] = 2		
		self.t['6-2'] = 2
		self.t['7-2'] = 2
		self.t['8-2'] = 2
		self.t['9-2'] = 2
		self.t['10-2'] = 2	
		self.t['1-3'] = 3
		self.t['2-3'] = 3
		self.t['3-3'] = 3
		self.t['4-3'] = 3
		self.t['5-3'] = 3
		self.t['6-3'] = 3
		self.t['7-3'] = 3
		self.t['8-3'] = 3
		self.t['9-3'] = 3
		self.t['10-3'] = 3
		self.t['1-4'] = 4
		self.t['2-4'] = 4
		self.t['3-4'] = 4
		self.t['4-4'] = 4
		self.t['5-4'] = 4
		self.t['6-4'] = 4
		self.t['7-4'] = 4
		self.t['8-4'] = 4
		self.t['9-4'] = 4
		self.t['10-4'] = 4
		self.t['1-5'] = 5
		self.t['2-5'] = 5
		self.t['3-5'] = 5
		self.t['4-5'] = 5
		self.t['5-5'] = 5
		self.t['6-5'] = 5
		self.t['7-5'] = 5
		self.t['8-5'] = 5
		self.t['9-5'] = 5
		self.t['10-5'] = 5
		self.t['1-6'] = 6	
		self.t['2-6'] = 6
		self.t['3-6'] = 6	
		self.t['4-6'] = 6	
		self.t['5-6'] = 6
		self.t['6-6'] = 6
		self.t['7-6'] = 6
		self.t['8-6'] = 6
		self.t['9-6'] = 6
		self.t['10-6'] = 6
		self.t['1-7'] = 7
		self.t['2-7'] = 7
		self.t['3-7'] = 7
		self.t['4-7'] = 7
		self.t['5-7'] = 7
		self.t['6-7'] = 7
		self.t['7-7'] = 7
		self.t['8-7'] = 7
		self.t['9-7'] = 7
		self.t['10-7'] = 7
		self.t['1-8'] = 8
		self.t['2-8'] = 8
		self.t['3-8'] = 8
		self.t['4-8'] = 8
		self.t['5-8'] = 8
		self.t['6-8'] = 8
		self.t['7-8'] = 8
		self.t['8-8'] = 8
		self.t['9-8'] = 8
		self.t['10-8'] = 8
		self.t['1-9'] = 9
		self.t['2-9'] = 9
		self.t['3-9'] = 9
		self.t['4-9'] = 9
		self.t['5-9'] = 9
		self.t['6-9'] = 9
		self.t['7-9'] = 9
		self.t['8-9'] = 9
		self.t['9-9'] = 9	
		self.t['10-9'] = 9
		self.t['1-10'] = 10
		self.t['2-10'] = 10
		self.t['3-10'] = 10
		self.t['4-10'] = 10
		self.t['5-10'] = 10
		self.t['6-10'] = 10
		self.t['7-10'] = 10
		self.t['8-10'] = 10
		self.t['9-10'] = 10
		self.t['10-10'] = 10

		
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
		screen_height = 800
		
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			#创建网格世界
			#self.line1 = rendering.Line((100,300),(500,300))
            		#创建信道模型1
			#self.chnl[0] = rendering.Line((100,300),(500,300))
			self.chnl1 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(200,400))
			self.chnl1.add_attr(self.circletrans)
			if self.terminate_states[1]:
				self.chnl1.set_color(0,255,0)        #available - green
			else:
				self.chnl1.set_color(255,0,0)        #blocked - red	

            		#创建第二个信道
			self.chnl2 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(400,400))
			self.chnl2.add_attr(self.circletrans)
			if self.terminate_states[2]:
				self.chnl2.set_color(0,255,0)        #available - green
			else:
				self.chnl2.set_color(255,0,0)        #blocked - red
			
			self.chnl3 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(600,400))
			self.chnl3.add_attr(self.circletrans)
			if self.terminate_states[3]:
				self.chnl3.set_color(0,255,0)        #available - green
			else:
				self.chnl3.set_color(255,0,0)        #blocked - red
			
			self.chnl4 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(800,400))
			self.chnl4.add_attr(self.circletrans)
			if self.terminate_states[4]:
				self.chnl4.set_color(0,255,0)        #available - green
			else:
				self.chnl4.set_color(255,0,0)        #blocked - red
			
			self.chnl5 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(1000,400))
			self.chnl5.add_attr(self.circletrans)
			if self.terminate_states[5]:
				self.chnl5.set_color(0,255,0)        #available - green
			else:
				self.chnl5.set_color(255,0,0)        #blocked - red
			
			self.chnl6 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(200,200))
			self.chnl6.add_attr(self.circletrans)
			if self.terminate_states[6]:
				self.chnl6.set_color(0,255,0)        #available - green
			else:
				self.chnl6.set_color(255,0,0)        #blocked - red
			
			self.chnl7 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(400,200))
			self.chnl7.add_attr(self.circletrans)
			if self.terminate_states[7]:
				self.chnl7.set_color(0,255,0)        #available - green
			else:
				self.chnl7.set_color(255,0,0)        #blocked - red
			
			self.chnl8 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(600,200))
			self.chnl8.add_attr(self.circletrans)
			if self.terminate_states[8]:
				self.chnl8.set_color(0,255,0)        #available - green
			else:
				self.chnl8.set_color(255,0,0)        #blocked - red
			
			self.chnl9 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(800,200))
			self.chnl9.add_attr(self.circletrans)
			if self.terminate_states[9]:
				self.chnl9.set_color(0,255,0)        #available - green
			else:
				self.chnl9.set_color(255,0,0)        #blocked - red
			
			self.chnl10 = rendering.make_circle(50)
			self.circletrans = rendering.Transform(translation=(1000,200))
			self.chnl10.add_attr(self.circletrans)
			if self.terminate_states[10]:
				self.chnl10.set_color(0,255,0)        #available - green
			else:
				self.chnl10.set_color(255,0,0)        #blocked - red
			
            		#创建机器人
			self.robot= rendering.make_circle(30)
			self.robotrans = rendering.Transform()
			self.robot.add_attr(self.robotrans)
			self.robot.set_color(255, 255, 255)
			
			self.viewer.add_geom(self.chnl1)
			self.viewer.add_geom(self.chnl2)
			self.viewer.add_geom(self.chnl3)
			self.viewer.add_geom(self.chnl4)
			self.viewer.add_geom(self.chnl5)
			self.viewer.add_geom(self.chnl6)
			self.viewer.add_geom(self.chnl7)
			self.viewer.add_geom(self.chnl8)
			self.viewer.add_geom(self.chnl9)
			self.viewer.add_geom(self.chnl10)
			self.viewer.add_geom(self.robot)

		else:
			if self.terminate_states[1]:
				self.chnl1.set_color(0,255,0)        #available - green
			else:
				self.chnl1.set_color(255,0,0)        #blocked - red

			if self.terminate_states[2]:
				self.chnl2.set_color(0,255,0)        #available - green
			else:
				self.chnl2.set_color(255,0,0)        #blocked - red

			if self.terminate_states[3]:
				self.chnl3.set_color(0,255,0)        #available - green
			else:
				self.chnl3.set_color(255,0,0)        #blocked - red

			if self.terminate_states[4]:
				self.chnl4.set_color(0,255,0)        #available - green
			else:
				self.chnl4.set_color(255,0,0)        #blocked - red

			if self.terminate_states[5]:
				self.chnl5.set_color(0,255,0)        #available - green
			else:
				self.chnl5.set_color(255,0,0)        #blocked - red

			if self.terminate_states[6]:
				self.chnl6.set_color(0,255,0)        #available - green
			else:
				self.chnl6.set_color(255,0,0)        #blocked - red

			if self.terminate_states[7]:
				self.chnl7.set_color(0,255,0)        #available - green
			else:
				self.chnl7.set_color(255,0,0)        #blocked - red

			if self.terminate_states[8]:
				self.chnl8.set_color(0,255,0)        #available - green
			else:
				self.chnl8.set_color(255,0,0)        #blocked - red

			if self.terminate_states[9]:
				self.chnl9.set_color(0,255,0)        #available - green
			else:
				self.chnl9.set_color(255,0,0)        #blocked - red

			if self.terminate_states[10]:
				self.chnl10.set_color(0,255,0)        #available - green
			else:
				self.chnl10.set_color(255,0,0)        #blocked - red
			
			
		
		if self.state is None: return None
		#self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
		self.robotrans.set_translation(self.x[self.state-1], self.y[self.state-1])
		
		
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

		
        
		
		
	
		
