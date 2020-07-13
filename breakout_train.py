import numpy as np
import gym
from breakout_ddqn import Agent
from collections import deque
import tensorflow as tf

agent = Agent(3)

num_episodes = 25000
env = gym.make('Breakout-v0')
def preprocess(state):
	state = tf.image.rgb_to_grayscale(state)
	state = tf.image.crop_to_bounding_box(state,34,0,160,160)
	state = tf.image.resize(state,[80,80])
	state = tf.squeeze(state)
	return state


rewards = []
for i in range(num_episodes):
	state = env.reset()
	state  = preprocess(state)
	# state = np.expand_dims(state,2)
	state = np.stack([state]*4,axis=2)
	# state = np.expand_dims(state,axis=0)
	# print('state done')

	tot_reward = 0
	done = False
	n_steps = 0
	z = 0
	while not done:
		action = agent.choose_action(state)
		action += 1
		# print('action got')
		new_state,reward,done,_ = env.step(action)

		tot_reward += reward
		new_state = preprocess(new_state)
		new_state = np.append(state[:,:,1:],np.expand_dims(new_state,2),axis=2)
		# new_state = np.expand_dims(new_state,axis=0)
		# print('new state processed')
		action -= 1
		agent.save_mem(state,action, reward,new_state,done)

		state = new_state
		n_steps += 1
		z += 1
		if done:
			break
		# print('rewards saved')

		if n_steps%4==0:
			agent.train()

	if i+1%10 == 0 and i != 0:
		agent.save_model()
		print('.... model saved ....')
		print(f'itteration: {i+1} \t reward: {tot_reward} \t average reward: {np.mean(rewards)}')

	else:
		if agent.epsilon > agent.min_epsilon:
			agent.epsilon -= agent.decay
		rewards.append(tot_reward)
		agent.align_models()
		print(f'itteration done: {i+1} with reward: {tot_reward}')

