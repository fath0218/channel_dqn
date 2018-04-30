import random, numpy, math, gym
import matplotlib.pyplot as plt

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import * 

#plt.ion()

class Brain:

	def __init__(self, stateCnt, actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt

		self.model = self._createModel()
		# self.model.load_weights("cartpole-basic.h5")

	def _createModel(self):
		model = Sequential()

		model.add(Dense(output_dim=64, activation='relu', input_dim=1))#stateCnt
		model.add(Dense(output_dim=actionCnt, activation='linear'))

		opt = RMSprop(lr=0.00025)
		model.compile(loss='mse', optimizer=opt)

		return model

	def train(self, x, y, epoch=1, verbose=0):
		self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

	def predict(self, s):
		return self.model.predict(s)

	def predict(self, s, target=False):
		if target:
			return self.model_.predict(s)
		else:
			return self.model.predict(s)

	def predictOne(self, s, target=False):
		return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()
		#return self.predict(self.stateCnt, target=target).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
	samples = []

	def __init__(self, capacity):
		self.capacity = capacity

	def add(self, sample):
		self.samples.append(sample)        

		if len(self.samples) > self.capacity:
			self.samples.pop(0)

	def sample(self, n):
		n = min(n, len(self.samples))
		return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
	steps = 0
	epsilon = MAX_EPSILON

	def __init__(self, stateCnt, actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt

		self.brain = Brain(stateCnt, actionCnt)
		self.memory = Memory(MEMORY_CAPACITY)
        
	def act(self, s):
		if random.random() < self.epsilon:
			print("*****random step*****")
			return random.randint(0, self.actionCnt-1)
		else:
			print("*****predict step*****")
			return numpy.argmax(self.brain.predictOne(s))

	def observe(self, sample):  # in (s, a, r, s_) format
		self.memory.add(sample)        

		# slowly decrease Epsilon based on our eperience
		self.steps += 1
		self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

	def replay(self):    
		batch = self.memory.sample(BATCH_SIZE)
		batchLen = len(batch)

		no_state = numpy.zeros(self.stateCnt)

		states = numpy.array([ o[0] for o in batch ])
		states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

		p = self.brain.predict(states)
		p_ = self.brain.predict(states_)

		x = numpy.zeros((batchLen, self.stateCnt))
		y = numpy.zeros((batchLen, self.actionCnt))
        
		for i in range(batchLen):
			o = batch[i]
			s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
			t = p[i]
			if s_ is None:
				t[a] = r
			else:
				t[a] = r + GAMMA * numpy.amax(p_[i])

			x[i] = s
			y[i] = t

		self.brain.train(x, y)

#-------------------- ENVIRONMENT ---------------------
class Environment:
	def __init__(self, problem):
		self.problem = problem
		self.env = gym.make(problem)
		self.channel_cnt = self.env.env.channel_cnt
		self.pick_times = [0 for x in range(self.channel_cnt)]
		self.overall_step = 0.000
		self.hundred_step = [0.000 for x in range(0, 100)]
		self.overall_connect = 0.000
		self.overall_one_step = 0.000
		self.hundred_one_step = [0.000 for x in range(0, 100)] 
		self.avg_step = 0
		self.avg_step_100 = 0

	def run(self, agent):
		s = self.env.reset()
		R = 0 
		step = 0                       #sum of steps in run_time period
		run_time = 0
        
		single_step = 0                #step for every successful try

		while True:     
			self.env.render()

			a = agent.act(s)
			step += 1
			single_step += 1

			print("act------------:", a+1)
			s_, r, done, info = self.env.step(a+1)
			#self.env.env.getRewardChart()
			#self.env.env.getTChart()
			print("reward:", r)

			if done: # terminal state
				self.pick_times[s_-1-self.channel_cnt] += 1
				run_time += 1
				if (single_step == 1):
					self.overall_one_step += 1
				single_step = 0 
				#s_ = None           # it's ok to run without this line, almost no influence to performance
				#agent.observe( (s, a, r, s_) )
				#s_ = self.env.reset()
				print ("done\n")

			agent.observe( (s, a, r, s_) ) #include add to memory
			agent.replay()            

			s = s_
			R += r

#			if done:
#				break

			if (run_time == 10):   #modifiable, output frequency
				break
	
		print("Total reward:", R)
		print("Steps taken:", step)
		self.hundred_step[int(self.overall_connect % 100)] = step
		self.overall_step += float(step)
 
#		if (step == 1):
#			self.overall_one_step += 1
#			self.hundred_one_step[int(self.overall_connect % 100)] = 1
#		else:
#			self.hundred_one_step[int(self.overall_connect % 100)] = 0
		
		self.overall_connect += 10

		avg_step = float(self.overall_step/self.overall_connect)
		print("Average steps:\t\t\t\t", avg_step)
		if int(self.overall_connect) > 100:
			avg_step_100 = float(sum(self.hundred_step)/100)
		else:
			avg_step_100 = float(sum(self.hundred_step)/int(self.overall_connect))
		print("Average steps of latest 100 tries:\t", avg_step_100)
       
		for i in range (self.channel_cnt):
			if self.pick_times[i] != 0:
				print("Channel %d picked times: %d" %(i+1, self.pick_times[i]))

		avg_one_step = float(self.overall_one_step/self.overall_connect)
		print("Overall success rate:\t\t\t%", avg_one_step * 100)
#		if int(self.overall_connect) > 100:
#			avg_one_step_100 = float(sum(self.hundred_one_step)/100)
#		else:
#			avg_one_step_100 = float(sum(self.hundred_one_step)/int(self.overall_connect))
#		print("Success rate of latest 100 tries:\t%", avg_one_step_100 * 100)

		print("Overall run time:", int(self.overall_connect))
		print("\n")
##############################################################################   CHART PART
#		#if int(self.overall_connect) < 245:
		x_scale = numpy.log(int(self.overall_connect))
#		#else: 
#		#    x_scale = numpy.log10(int(self.overall_connect)) + 3.1
		plt.subplot(2,1,1)
#		plt.axis()
		if int(self.overall_connect) < 245:
			plt.scatter(x_scale, avg_step, c = 'r')
			plt.scatter(x_scale, avg_step_100, c = 'b')
			plt.pause(0.00001)
		elif (int(self.overall_connect) < 1000) and (int(self.overall_connect) % 10 == 0):
			plt.scatter(x_scale, avg_step, c = 'r')
			plt.scatter(x_scale, avg_step_100, c = 'b')
			plt.pause(0.00001)
		elif (int(self.overall_connect) < 2000) and (int(self.overall_connect) % 30 == 0):
			plt.scatter(x_scale, avg_step, c = 'r')
			plt.scatter(x_scale, avg_step_100, c = 'b')
			plt.pause(0.00001)
		elif (int(self.overall_connect) < 30000) and (int(self.overall_connect) % 100 == 0):
			plt.scatter(x_scale, avg_step, c = 'r')
			plt.scatter(x_scale, avg_step_100, c = 'b')
			plt.pause(0.00001)
        
		plt.subplot(2,1,2)
		if (int(self.overall_connect) % 100 == 0):
			plt.bar(range(len(self.pick_times)), self.pick_times, color="blue")

#-------------------- MAIN ----------------------------
PROBLEM = 'Channel-v0'
env = Environment(PROBLEM)

stateCnt  = np.array(1)#env.env.observation_space.shape[0]
actionCnt = env.env.env.channel_cnt

agent = Agent(stateCnt, actionCnt)

env.run(agent)
env.run(agent)
env.run(agent)
try:
	while True:
		env.run(agent)
finally:
	agent.brain.model.save("channel.h5")
