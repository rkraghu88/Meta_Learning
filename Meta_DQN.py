import random
import numpy as np
# import gym
from collections import deque
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.callbacks import History
from statistics import mean
import AdamOpt as AdamOpt
import AdamOpt as AdamOptMeta
import SharedWeights
import math

# import pickle

import tensorflow as tf
import NeuralApprox as NA

#tf.disable_v2_behavior()
history = History()

Episodes = 1000


class DQNAgent:
    def __init__(self, state_size, action_size, power_constraint, power_val_array, meta_param_len,id):
        self.agent_id=id
        self.state_size = state_size
        self.action_size = action_size
        self.power_constraint = power_constraint
        self.power_val_array = np.array(power_val_array)
        self.power_val_chosen = []
        self.mean_pow_val = 0
        self.mean_pow_val_check = 0
        self.penalty_lambda = 1 / np.amax(self.power_val_array)
        self.lambda_learning_rate = 0.0001
        self.lambda_lr_decay = 0.99993
        self.penalty_lambda_array = []
        self.penalty_lambda_array = np.array(self.penalty_lambda_array)
        self.AdamOpt=AdamOpt.AdamOpt(step=self.lambda_learning_rate)
        self.AdamOptMeta = AdamOptMeta.AdamOpt(step=self.lambda_learning_rate,sign=-1)
        self.memory = deque(maxlen=30000)  # Memory D for storing states, actions, rewards etc
        self.meta_memory = deque(maxlen=1000)  # Memory D for storing states, actions, rewards etc
        self.gamma = 0.9  # discount factor gamma = 1 (average case)
        self.epsilon = 1.0  # keep choosing random actions in the beginning and decay epsilon as time
        # progresses
        self.epsilon_min = 0.1  # minimum exploration rate
        self.epsilon_decay = 0.98  # decay rate. (epsilon = epsilon * epsilon_decay)
        self.learning_rate = 0.001  # learning rate for optimizer in neural network
        self.batch_size = 64  # mini batch size for replay

        self.model = self.build_model()  # neural network to learn q function
        self.target_model = self.build_model()  # neural network to estimate target q function
        self.meta_model = self.build_model()

        self.update_target_model()  # Initialize target model to be same as model (theta_ = theta)
        # self.power_values = np.arange(1, 52, 2.55) / 49.45
        self.target_model_update_count = 0
        self.cumulative_reward = 0
        self.average_reward = 0
        self.num_of_actions = 0
        self.reward_array = []
        self.reward_array = np.array(self.reward_array)
        self.meta_param_len=meta_param_len
        self.DSGDA=NA.DNNApproximator((1,self.meta_param_len),1,.01,.01)
        # SharedWeights.weights = np.append(SharedWeights.weights, self.target_model.get_weights())
        SharedWeights.weights.append(self.target_model.get_weights())
        SharedWeights.weight_size=SharedWeights.weight_size+1
        # self.update_global_weight()
        # self.get_meta_actor_weight(np.ones(SharedWeights.weight_size)*.1)
        #SharedWeights.weight_size=np.size(self.meta_model.get_weights())
    # This function builds a neural network consisting of 4 layers including the input layer
    def build_model(self):
        model = Sequential()
        # Input layer and hidden layer 1. kernel_initializer gives random values to weights according to specified dist.
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        # Hidden layer 2
        model.add(Dense(64, activation='relu'))
        # Output layer
        model.add(Dense(self.action_size, activation='relu'))

        # Compile the model
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=0.00001))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=0.0))
        return model

    # This function sets weights of target model to be same as model used for training
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_global_weight(self):
        # print(np.shape(SharedWeights.weights))
        # print(np.shape(self.target_model.get_weights()))
        # print(np.shape(self.target_model.get_weights()[0]))
        # print(np.shape(self.target_model.get_weights()[1]))
        # print(np.shape(self.target_model.get_weights()[2]))
        # print(np.shape(self.target_model.get_weights()[3]))
        # print(np.shape(self.target_model.get_weights()[4]))
        # print(np.shape(self.target_model.get_weights()[5]))
        print("Agent %d Updating Global Weights..."%self.agent_id)
        SharedWeights.weights[self.agent_id]=self.target_model.get_weights()
        # print(len(SharedWeights.weights))


    def get_meta_actor_weight(self, meta_param):
        temp_weight = SharedWeights.weights[0]

        for j in range(len(temp_weight)):
            temp_weight[j]= 0.0*SharedWeights.weights[0][j]
        # print(temp_weight)
        for i in np.arange(0,len(meta_param)):
            for j in range(len(temp_weight)):
                #print(meta_param)
                temp_weight[j]=temp_weight[j]+meta_param[i]*SharedWeights.weights[i][j]
        #print('Length:',len(temp_weight))
        return temp_weight

    def update_meta_actor(self, weights):
        self.meta_model.set_weights(weights)

    # Get action using epsilon greedy policy

    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            self.num_of_actions = self.num_of_actions + 1
            action_power_index = random.randrange(self.action_size)  # random action
            self.power_val_chosen.append(self.power_val_array[action_power_index])
            self.mean_pow_val = mean(self.power_val_chosen)
            return action_power_index
        else:
            state = np.reshape(state, [1, self.state_size])
            q_value = self.model.predict(state)
            max_reward_value = np.amax(q_value[0])
            action_power_index = np.argmax(q_value[0])  # action giving max reward
            self.power_val_chosen.append(self.power_val_array[action_power_index])
            self.mean_pow_val = mean(self.power_val_chosen)

            self.cumulative_reward = self.cumulative_reward + max_reward_value
            self.num_of_actions = self.num_of_actions + 1
            self.average_reward = self.cumulative_reward / self.num_of_actions
            self.reward_array = np.append(self.reward_array, max_reward_value)
            return action_power_index  # choose action (power value)which gives maximum reward.

    def get_meta_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     self.num_of_actions = self.num_of_actions + 1
        #     action_power_index = random.randrange(self.action_size)  # random action
        #     self.power_val_chosen.append(self.power_val_array[action_power_index])
        #     self.mean_pow_val = mean(self.power_val_chosen)
        #     return action_power_index
        # else:
        state = np.reshape(state, [1, self.state_size])
        q_value = self.meta_model.predict(state)
        max_reward_value = np.amax(q_value[0])
        action_power_index = np.argmax(q_value[0])  # action giving max reward
        self.power_val_chosen.append(self.power_val_array[action_power_index])
        self.mean_pow_val = mean(self.power_val_chosen)

        self.cumulative_reward = self.cumulative_reward + max_reward_value
        self.num_of_actions = self.num_of_actions + 1
        self.average_reward = self.cumulative_reward / self.num_of_actions
        self.reward_array = np.append(self.reward_array, max_reward_value)
        return action_power_index  # choose action (power value)which gives maximum reward.

    def meta_step(self, meta_param):
        meta_param=self.AdamOptMeta.AdamOptimizer(meta_param,self.DSGDA.gradient_function(meta_param),1.0)
        return meta_param[0][0]

    # def meta_train(self,inputs,outputs):
    #     self.DSGDA.train_on_batch(inputs,outputs)
    # save sample in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # append to memory

    def meta_remember(self,input,output):
        self.meta_memory.append((input,output))
    # Replay from memory

    def meta_replay(self,batch_size):
        mini_batch=random.sample(self.meta_memory,batch_size)
        input_memory=deque()
        output_memory=deque()
        for input_i,output_i in mini_batch:
            input_memory.append(input_i)
            output_memory.append(output_i)
        output = (np.array(output_memory)).reshape(batch_size, 1, 1)
        input = (np.array(input_memory)).reshape(batch_size, 1, self.meta_param_len)
        self.DSGDA.train_on_batch(input,output)

    def penalty_step(self):
        self.mean_pow_val = mean(self.power_val_chosen[-200:])
        self.penalty_lambda = self.AdamOpt.AdamOptimizer(self.penalty_lambda,
                                                         (self.mean_pow_val - self.power_constraint), 1)
        # self.penalty_lambda=self.AdamOpt.AdamOptimizer(self.penalty_lambda,(self.mean_pow_val - self.power_constraint),1/(1+0.00001*self.target_model_update_count*np.log10(10+np.log10(10+np.log10(10+self.target_model_update_count)))))
        # self.penalty_lambda = self.penalty_lambda + self.lambda_learning_rate * (self.mean_pow_val - self.power_constraint)
        self.penalty_lambda = max(self.penalty_lambda, 1 / np.amax(self.power_val_array))
        # self.lambda_learning_rate = self.lambda_learning_rate * self.lambda_lr_decay
        # self.lambda_learning_rate = self.lambda_learning_rate * 1/(1+0.00001*self.target_model_update_count*np.log10(10+np.log10(10+self.target_model_update_count)))
        self.penalty_lambda_array = np.append(self.penalty_lambda_array, self.penalty_lambda)
    def replay(self):
        batch_size = min(len(self.memory), self.batch_size)
        mini_batch = random.sample(self.memory, batch_size)  # sample from memory and create a mini batch
        state_array = []
        state_array = np.array(state_array)
        target_array = []
        target_array = np.array(target_array)
        for state, action, reward, next_state, done in mini_batch:
            state = np.reshape(state, [1, self.state_size])
            state_array = np.append(state_array, state)
            next_state = np.reshape(next_state, [1, self.state_size])
            q_value = self.target_model.predict(next_state)
            max_reward_value = np.amax(q_value[0])  # max reward for next state (max_a'(Q^(s',a',theta^)))
            target = (reward - self.penalty_lambda * self.power_val_array[action] + self.gamma * max_reward_value)
            # y_j + gamma *max_a'(Q^(s',a',theta^))
            # try:
            #     with tf.device("gpu:0"):
            #         target_f = self.model.predict(state)
            # except:
            target_f = self.model.predict(state)
            target_f[0][action] = target
            target_array = np.append(target_array, target_f)

        state_array = np.reshape(state_array, [batch_size, self.state_size])
        target_array = np.reshape(target_array, [batch_size, self.action_size])
        # try:
        #     with tf.device("gpu:0"):
        #         hist = self.model.fit(state_array, target_array, epochs=1, verbose=0)  # run neural network and back propagation
        # except:
        hist = self.model.fit(state_array, target_array, epochs=1, verbose=0)  # run neural network and back propagation
        # with open('/home/pratheek/IISc_ML/Machine Learning/Neuralnets/Ram '
        #        'simulation/copy_multiQueue/lam0.1_abs.error_epochs10_q>max', 'wb') as file_pi:
        #  pickle.dump(hist.history, file_pi)
        # self.network_loss_array = np.append(self.network_loss_array, hist.history['loss'])

        self.target_model_update_count = self.target_model_update_count + 1
        if (self.target_model_update_count % 100) == 0:
            self.update_target_model()

        if (self.target_model_update_count % 1) == 0:
            self.penalty_step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # reducing epsilon


# Not required
''' if __name__ == "__main__":
       state_size = 4
       action_size = 5

       agent = DQNAgent(state_size, action_size)
       done = False

       for e in range(Episodes):
           state = env.reset()
           state = np.reshape(state, [1, state_size])
           for time in range(500):
               # env.render()
               action = agent.get_action(state)
               next_state, reward, done, _ = env.step(action)
               reward = reward if not done else -10
               next_state = np.reshape(next_state, [1, state_size])
               agent.append_sample(state, action, reward, next_state, done)
               state = next_state
               if done:
                   print("episode: {}/{}, score: {}, e: {:.2}"
                         .format(e, Episodes, time, agent.epsilon))
                   break
               if len(agent.memory) > batch_size:
                   agent.replay(batch_size)
           '''
