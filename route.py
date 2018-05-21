import random, numpy, math, gym
import matplotlib.pyplot as plt

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import routeConfig as routeConfig

#plt.ion()
global literation           #Number of test literation

OBSERV_BATCH = routeConfig.OBSERV_BATCH
USR_CNT = routeConfig.USR_CNT

class Brain:

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=USR_CNT))#stateCnt
        model.add(Dense(output_dim=64, activation='relu'))          #######################################
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
        #return self.predict(s.reshape(1, stateCnt), target=target).flatten()

        print ("**************################", np.array(s))
        return self.predict(np.array(s).reshape(1, USR_CNT), target=target).flatten()############################

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)
        #print ("samples\n\n",self.samples)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 256

GAMMA = 0.7

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.00025      # speed of decay

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
        #print (batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        #x = numpy.zeros((batchLen, self.stateCnt))
        x = numpy.zeros((batchLen, np.array(USR_CNT))) #######################################
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
        self.total_path_cnt = self.env.env.total_path_cnt
        self.total_abc_path_cnt = self.env.env.path_a_cnt + self.env.env.path_b_cnt + self.env.env.path_c_cnt
        self.pick_times = [0 for x in range(self.total_path_cnt)]
        self.overall_step = 0.000
        self.hundred_step = [0.000 for x in range(0, 100)]
        self.overall_connect = 0.000
        self.overall_one_step = 0.000
        self.hundred_one_step = [0.000 for x in range(0, 100)]
        self.avg_step = 0
        self.avg_step_100 = 0

    def run(self, agent):
        s = self.env.reset()
        print ("s by reset in run()", s)
        R = 0
        step = 0                       #sum of steps in run_time period
        run_time = 0
        global literation           #Number of test literation

        single_step = 0                #step for every successful try
        f_agent = open("agent.csv", "w")
        f_agent.write('state,,,action,reward,next_state\n')
        f_channel = open("path_available.csv", "w")
        for i in range(self.total_path_cnt):
            f_channel.write(str(i+1))
            if i < self.total_path_cnt-1:
                f_channel.write(',')
            else:
                f_channel.write('\n')

        state_batch = [0 for x in range(0, OBSERV_BATCH)]  
        next_state_batch = [0 for x in range(0, OBSERV_BATCH)]  

        state_batch[0] = s

        for i in range(0, literation):
            self.env.render()

            a = agent.act(s) 
            #a = agent.act(state_batch)

            step += 1
            single_step += 1

            print("act------------:", a+1)
            #self.env.env.setStateBatch(state_batch)#################################################
            
            s_, r, done, info = self.env.step(a+1)
            print ("s_ by step", s_)

            path_available = self.env.env.getChannelAvailable()
            #self.env.env.getRewardChart()
            #self.env.env.getTChart()
            print("reward:", r)

           # if done: # terminal state
           #     self.pick_times[s_-1-self.channel_cnt] += 1
           #     run_time += 1
           #     if (single_step == 1):
           #         self.overall_one_step += 1
           #     single_step = 0
           #     #s_ = None           # it's ok to run without this line, almost no influence to performance
           #     #agent.observe( (s, a, r, s_) )
           #     #s_ = self.env.reset()
           #     print ("done\n")

            for i in range(OBSERV_BATCH-1, 0, -1): 
                next_state_batch[i] = state_batch[i-1]

            next_state_batch[0]= s_

            #agent.observe( (s, a, r, s_) ) #include add to memory
            #print ( "stuff to be observed#############",(np.array(s), a, r, np.array(s_)) )
            #agent.observe( (np.array(s), a, r, np.array(s_)) )
            print ( s_ )
            print ( "stuff to be observed#############",(np.array(s), a, r, np.array(s_)) )
            agent.observe( (np.array(s), a, r, np.array(s_)) )
           #agent.observe( (np.array(state_batch), a, r, np.array(next_state_batch)) )###################################

            agent.replay()


            record_str = str(s[0])+','+str(s[1])+','+str(s[2])+','+str(a)+','+str(r)+','+str(s_[0])+','+str(s_[1])+','+str(s_[2])+'\n'
            f_agent.write(record_str)
            for j in range(self.total_path_cnt):
                f_channel.write(str(path_available[j+1]))
                if j < self.total_path_cnt-1:
                    f_channel.write(',')
                else:
                    f_channel.write('\n')

            s = s_
            for i in range(0, OBSERV_BATCH):   
                state_batch[i] = next_state_batch[i]

            R += r

        f_agent.close()
        f_channel.close()

    def draw_plots(self):
        global literation
        f_agent = open("agent.csv", "r")
        f_agent.readline()
        path_chosen_cnt = [0] * self.total_abc_path_cnt # length 11
        success_connexion = 0
        block_num = 30
        block_cnt = 0
        local_ind_ls = []
        local_success_ls = []
        local_success = 0
        local_success_pure_ls = []
        local_success_pure = 0
        chnl = self.env.env.channel_p1


        for i in range(0, literation):
            step_str = f_agent.readline()
            s = [0,0,0]
            s[0] = int(step_str.split(",")[0]) # Read the current state
            s[1] = int(step_str.split(",")[1])
            s[2] = int(step_str.split(",")[2])
            path_chosen_cnt[self.env.env.getChannelNumber(s[0],'a') - 1] += 1
            path_chosen_cnt[self.env.env.getChannelNumber(s[1],'b') - 1 + 2] += 1
            path_chosen_cnt[self.env.env.getChannelNumber(s[2],'c') - 1 + 4] += 1

            if not self.env.env.isChannelBlocked(s[0],'a'): # success connexion
                success_connexion += 1
                local_success += 1
                if not self.env.env.isChannelOccupied(s[0],'a'):
                    local_success_pure += 1
            if not self.env.env.isChannelBlocked(s[1],'b'): # success connexion
                success_connexion += 1
                local_success += 1
                if not self.env.env.isChannelOccupied(s[1],'b'):
                    local_success_pure += 1
            if not self.env.env.isChannelBlocked(s[2],'c'): # success connexion
                success_connexion += 1
                local_success += 1
                if not self.env.env.isChannelOccupied(s[2],'c'):
                    local_success_pure += 1

            block_cnt += 1
            if block_cnt % block_num == 0:
                # count the success number of block_num communications
                block_cnt = 0
                local_ind_ls.append(i)
                local_success_ls.append((local_success / block_num) / 3)
                local_success_pure_ls.append((local_success_pure / block_num) / 3)
                local_success = 0
                local_success_pure = 0
                block_cnt = 0

        path_chosen_cnt_a = [0] * self.env.env.path_a_cnt
        path_chosen_cnt_b = [0] * self.env.env.path_b_cnt
        path_chosen_cnt_c = [0] * self.env.env.path_c_cnt
        path_chosen_cnt_sum = [0] * self.total_path_cnt

        path_chosen_cnt_a[0] = path_chosen_cnt[0]
        path_chosen_cnt_b[0] = path_chosen_cnt[3]
        path_chosen_cnt_c[0] = path_chosen_cnt[7]        
        path_chosen_cnt_a[1] = path_chosen_cnt[1]
        path_chosen_cnt_b[1] = path_chosen_cnt[4]
        path_chosen_cnt_c[1] = path_chosen_cnt[8]
        path_chosen_cnt_a[2] = path_chosen_cnt[2]
        path_chosen_cnt_b[2] = path_chosen_cnt[5]
        path_chosen_cnt_c[2] = path_chosen_cnt[9]
        path_chosen_cnt_b[3] = path_chosen_cnt[6]
        path_chosen_cnt_c[3] = path_chosen_cnt[10]

        path_chosen_cnt_sum[0] = path_chosen_cnt[0]
        path_chosen_cnt_sum[1] = path_chosen_cnt[1] + path_chosen_cnt[3]
        path_chosen_cnt_sum[2] = path_chosen_cnt[2] + path_chosen_cnt[4]
        path_chosen_cnt_sum[3] = path_chosen_cnt[5] + path_chosen_cnt[7]
        path_chosen_cnt_sum[4] = path_chosen_cnt[6] + path_chosen_cnt[8]
        path_chosen_cnt_sum[5] = path_chosen_cnt[9]
        path_chosen_cnt_sum[6] = path_chosen_cnt[10]

        random_rate = ( ((1-chnl[0])+(1-chnl[1])+(1-chnl[2]))/3 + ((1-chnl[4])+(1-chnl[3])+(1-chnl[1])+(1-chnl[2]))/4 + ((1-chnl[4])+(1-chnl[3])+(1-chnl[5])+(1-chnl[6]))/4 ) / 3
        optimal_rate = ((1-chnl[0])*(1-chnl[0])+(1-chnl[1])*(1-chnl[1])+(1-chnl[2])*(1-chnl[2])) / ((1-chnl[0])+(1-chnl[1])+(1-chnl[2]))
        optimal_rate += ((1-chnl[3])*(1-chnl[3])+(1-chnl[1])*(1-chnl[1])+(1-chnl[2])*(1-chnl[2])+(1-chnl[4])*(1-chnl[4])) / ((1-chnl[4])+(1-chnl[1])+(1-chnl[2])+(1-chnl[3]))
        optimal_rate += ((1-chnl[3])*(1-chnl[3])+(1-chnl[5])*(1-chnl[5])+(1-chnl[6])*(1-chnl[6])+(1-chnl[4])*(1-chnl[4])) / ((1-chnl[4])+(1-chnl[5])+(1-chnl[6])+(1-chnl[3]))
        optimal_rate /= 3

        # Draw charts
        plt.subplot(2,2,1) # Draw outage probability
        plt.plot(local_ind_ls, local_success_ls) 
        plt.hlines(random_rate, 0, literation, colors = "c", linestyles = "dashed")
        plt.hlines(optimal_rate, 0, literation, colors = "c", linestyles = "dashed")
        plt.hlines((2.58/3), 0, literation, colors = "r", linestyles = "dashed")
        plt.ylabel('Success Rate')
        plt.subplot(2,2,2)
        plt.bar(range(1, len(path_chosen_cnt_a)+1), path_chosen_cnt_a, color="cyan", align='center')
        plt.bar(range(4, len(path_chosen_cnt_b)+4), path_chosen_cnt_b, color="blue", align='center')
        plt.bar(range(8, len(path_chosen_cnt_c)+8), path_chosen_cnt_c, color="magenta", align='center')
        plt.ylabel('Pick Time')
        plt.subplot(2,2,3)
        plt.plot(local_ind_ls, local_success_pure_ls)
        plt.ylabel('Success Rate Pure')
        plt.subplot(2,2,4)
        plt.bar(range(1, len(path_chosen_cnt_sum)+1), path_chosen_cnt_sum, color="orange", align='center')
        plt.ylabel('Pick Time')
        plt.show()

        f_agent.close()
#            if done:
#                break

    #        if (run_time == 10):   #modifiable, output frequency
    #            break

        #print("Total reward:", R)
        #print("Steps taken:", step)
        #self.hundred_step[int(self.overall_connect % 100)] = step
        #self.overall_step += float(step)

#        #if (step == 1):
#        #    self.overall_one_step += 1
#        #    self.hundred_one_step[int(self.overall_connect % 100)] = 1
#        #else:
#        #    self.hundred_one_step[int(self.overall_connect % 100)] = 0
        #
        #self.overall_connect += 10

        #avg_step = float(self.overall_step/self.overall_connect)
        #print("Average steps:\t\t\t\t", avg_step)
        #if int(self.overall_connect) > 100:
        #    avg_step_100 = float(sum(self.hundred_step)/100)
        #else:
        #    avg_step_100 = float(sum(self.hundred_step)/int(self.overall_connect))
        #print("Average steps of latest 100 tries:\t", avg_step_100)

        #for i in range (self.channel_cnt):
        #    if self.pick_times[i] != 0:
        #        print("Channel %d picked times: %d" %(i+1, self.pick_times[i]))

        #avg_one_step = float(self.overall_one_step/self.overall_connect)
        #print("Overall success rate:\t\t\t%", avg_one_step * 100)
#        #if int(self.overall_connect) > 100:
#        #    avg_one_step_100 = float(sum(self.hundred_one_step)/100)
#        #else:
#        #    avg_one_step_100 = float(sum(self.hundred_one_step)/int(self.overall_connect))
#        #print("Success rate of latest 100 tries:\t%", avg_one_step_100 * 100)

        #print("Overall run time:", int(self.overall_connect))
        #print("\n")
###############################################################################   CHART PART
#        ##if int(self.overall_connect) < 245:
        #x_scale = numpy.log(int(self.overall_connect))
#        ##else:
#        ##    x_scale = numpy.log10(int(self.overall_connect)) + 3.1
        #plt.subplot(2,1,1)
#        #plt.axis()
        #if int(self.overall_connect) < 245:
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)
        #elif (int(self.overall_connect) < 1000) and (int(self.overall_connect) % 10 == 0):
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)
        #elif (int(self.overall_connect) < 2000) and (int(self.overall_connect) % 30 == 0):
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)
        #elif (int(self.overall_connect) < 30000) and (int(self.overall_connect) % 100 == 0):
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)

        #plt.subplot(2,1,2)
        #if (int(self.overall_connect) % 100 == 0):
        #    plt.bar(range(len(self.pick_times)), self.pick_times, color="blue")

#-------------------- MAIN ----------------------------
global literation
literation = routeConfig.literation
PROBLEM = 'Route-v0'
env = Environment(PROBLEM)

stateCnt  = np.array(1)#env.env.observation_space.shape[0]
actionCnt = env.env.env.path_a_cnt * env.env.env.path_b_cnt * env.env.env.path_c_cnt

agent = Agent(stateCnt, actionCnt)

#env.run(agent)
#env.run(agent)
#env.run(agent)
try:
    #while True:
    env.run(agent)
    env.draw_plots()
finally:
    agent.brain.model.save("channel.h5")
