import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential,Model
from keras.layers import Dense, Reshape, Flatten, Activation,Input,merge

from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

import networkx as nx
import pylab as plt
import numpy as np
import logging.config


import random


class PGAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = 0.99
		self.learning_rate = 0.001
		self.states = []
		self.gradients = []
		self.rewards = []
		self.probs = []
		self.model = self._build_model() 
		self.model.summary()

	def _build_model(self):

		model = Sequential()
		model.add(Reshape((1, 2, 64), input_shape=(self.state_size,)))
		model.add(Convolution2D(2, 2, 4, subsample=(2, 2), border_mode='same',activation='relu', init='he_uniform'))
		model.add(Flatten())
		model.add(Dense(16, activation='relu', init='he_uniform'))
		model.add(Dense(8, activation='relu', init='he_uniform'))
		model.add(Dense(self.action_size, activation='softmax'))
		opt = Adam(lr=self.learning_rate)
		model.compile(loss='categorical_crossentropy', optimizer=opt)

		return model

	def remember(self, state, action, prob, reward):
		y = np.zeros([self.action_size])
		y[action] = 1
		self.gradients.append(np.array(y).astype('float32') - prob)
		self.states.append(state)
		self.rewards.append(reward)

	def act(self, state):

		state = state.reshape([1, state.shape[0]])


		
		#print("STATETOMODEL::",state)

		aprob = self.model.predict(state, batch_size=3).flatten()



		self.probs.append(aprob)
		#prob = aprob / np.sum(aprob)
		action = np.random.choice(self.action_size, 1, p=aprob)[0] #exploration and exploitation
		#[]

		#print('np.sum(aprob):', np.sum(aprob))
		#print("aprob::::", aprob)


		#print("ACTION:", action)

		return action, aprob

	def discount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, rewards.size)):
			if rewards[t] != 0:
				running_add = 0
			running_add = running_add * self.gamma + rewards[t]
			discounted_rewards[t] = running_add
		return discounted_rewards

	def train(self):
		gradients = np.vstack(self.gradients)
		rewards = np.vstack(self.rewards)
		rewards = self.discount_rewards(rewards)
		rewards = (rewards - np.mean(rewards)) / (np.std(rewards) - np.mean(rewards))
		 #normalize

		#print('rewards::', rewards)

		gradients *= rewards
		X = np.squeeze(np.vstack([self.states]))
		Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))



		self.model.train_on_batch(X, Y)
		self.states, self.probs, self.gradients, self.rewards = [], [], [], []

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)





class Environment:

	def __init__(self):
		self.__version__ = "0.1.0"
		logging.info("BananaEnv - Version {}".format(self.__version__))

		self.points_list=[(100,0),(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(4,9),
                (4,10),(5,11),(5,12),(6,13),(6,14),(7,15),(7,16), (8,17),
                (8,18),(9,19),(9,20),(10,21),(10,22),(11,23),(11,24),(12,25),
                (12,26), (13,27),(13,28),(14,29),(14,30),(15,31),(15,32),(16,33),
                (16,34),(17,35),(17,36),(18,37),(18,39),(19,40),(19,41),(20,42),(20,43),
                (21,44),(21,45),(22,46),(22,47),(23,48),(23,49),(24,50),(24,51),(25,52),(25,53),
                (26,54),(26,55),(27,56),(27,57),(28,58),(28,59),(29,60),(29,61),(30,62),(30,63)]

		self.attrs = { # all nodes have [1,0] label  

        0: {'attr': [[0,1]]}, 1: {'attr': [[0,1]]},
        2: {'attr': [[0,1]]}, 5: {'attr': [[0,1]]},  12: {'attr': [[0,1]]},25: {'attr': [[0,1]]},
        52: {'attr': [[0,1]]}, 6: {'attr': [[0,1]]}, 14: {'attr': [[0,1]]},  8: {'attr': [[0,1]]}, 27: {'attr': [[0,1]]}
        }

		self.G=nx.DiGraph()
		self.G.add_edges_from(self.points_list)

		self.labels = []
		nx.set_node_attributes(self.G, self.labels, 'attr')
		self.labels.append([1,0])

		nx.set_node_attributes(self.G, self.attrs)

		self.state = 100


		self.done = self.is_over() # True or False

		self.curr_episode = -1

		self.action_episode_memory = []
		self.reward_episode_memory = []



	def step(self, action): #action is 0 for left 1 for right

		self.do_action(action)#the state is updated 
		ob = self._get_state()
		reward = self._get_reward()
		done = self.is_over()

		return ob,reward,done


	def _neighbors(self):
		tmp = []
		for i in self.G.neighbors(self.state):
			tmp.append(i)
		return tmp 

	def do_action(self,action): #0 

		self.action_episode_memory[self.curr_episode].append(action)

		neighbors = self._neighbors() # [7,8]
		if len(neighbors) ==1:
			#print("Len neighbors: ", len(neighbors))
			self.state=0

		else:
			#print("Len neighbors: ", len(neighbors))
			self.state = neighbors[action] #neighbors[0] = 7





	def _get_reward(self):
		if self.G.nodes[self.state]['attr'][0] == [0,1]: 
			self.reward_episode_memory[self.curr_episode].append(0.2)
			return 0.2
		else:

			if self.G.nodes[self.state]['attr'][0] == [1,0]:
				self.reward_episode_memory[self.curr_episode].append(0.0)
				return 0
			else:
				raise Exception("Go to get reward something wrong with node label")



	def reset(self):
		self.state = 100
		self.curr_episode += 1 #-1 ==>0 for the first time
		self.action_episode_memory.append([])
		self.reward_episode_memory.append([])
		return self.state

	def is_over(self):
		neighbors = self._neighbors()
		if len(neighbors) == 0:
			return True
		else: 
			return False


	def _get_state(self):
		return self.state #just returns the state




if __name__ == "__main__":

	env = Environment()
	state = env.reset()
	score = 0
	episode = 0
	path = []
	x_data = []
	y_data = []

	state_size = 64 * 2 #number of nodes in graph
	action_size = 2



	agent = PGAgent(state_size, action_size)

	state, reward, done  = env.step(0)
	i = 0
	while i<10000:
		

		base = np.zeros((64,2))
		base[state][0] = 1

		#print("base::",base)

		action, prob = agent.act(base.astype(np.float).ravel())

		state, reward, done  = env.step(action)
		
		path.append(state)
		score += reward
		#print(done)

		base = np.zeros((64,2))
		base[state][0] = 1 # = [0,0,0,0,0,1,0,0,0,...]

		agent.remember(base.astype(np.float).ravel(), action, prob, reward)

		if done==True:
			i+=1
			episode += 1
			agent.train()
			print("path:", path)
			path = []
			print('Episode: %d - Score: %f.' % (episode, score))
			if i%10==0:
				x_data.append(score)
				y_data.append(episode)
			score = 0
			state = env.reset()
			state, reward, done  = env.step(0)


# Plot the data
	plt.plot(y_data, x_data, label='linear')
# Add a legend
	plt.legend()
# Show the plot
	plt.show()

