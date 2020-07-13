import gym
import tensorflow as tf
import numpy as np
from collections import deque
import os
from datetime import datetime
import random
	
class Agent:
	def __init__(self,num_actions,input_dim=(210,160,4),max_queue=1000,epsilon=1.0,
		decay=0.001,min_epsilon=0.01,gamma=0.90,batch_size=32):
		self.input_dim = input_dim
		self.num_actions = num_actions
		self.q_model = self.make_model()
		self.target_model = self.make_model()
		self.memory = deque(maxlen=max_queue)
		self.epsilon = epsilon
		self.gamma = gamma
		self.min_epsilon = min_epsilon
		self.decay = decay
		self.BATCH_SIZE = batch_size

	def save_mem(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def make_model(self):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Conv2D(64,3,activation='relu',input_shape=(80,80,4)))
		model.add(tf.keras.layers.MaxPooling2D((2,2)))
		model.add(tf.keras.layers.Conv2D(64,3,activation='relu'))
		model.add(tf.keras.layers.MaxPooling2D((2,2)))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(256,activation='relu'))
		model.add(tf.keras.layers.Dense(self.num_actions))

		model.compile(optimizer='adam',loss='mse')

		return model

	def align_models(self):
		self.target_model.set_weights(self.q_model.get_weights())

	def choose_action(self,state):
		if np.random.random() < self.epsilon:
			action = np.random.choice([i for i in range(self.num_actions)])
		else:
			q_vals = self.target_model.predict(np.expand_dims(state,0))
			action = np.argmax(q_vals)
		return action

	def train(self):
		if len(self.memory) < self.BATCH_SIZE:
			return

		sample = random.sample(self.memory,self.BATCH_SIZE)
		for state,action,reward,new_state,done in sample:
			target = self.q_model.predict(np.expand_dims(state,0))

			if done:
				target[0][action] = reward
			else:
				t = self.target_model.predict(np.expand_dims(new_state,0))
				target[0][action] = reward + self.gamma * np.max(t)
			self.q_model.fit(np.expand_dims(state,axis=0),target,epochs=1,verbose=0)


	def save_model(self):
		self.q_model.save(f'models/q_model.h5')
		self.target_model.save(f'models/target_model.h5')



