import gym
import numpy as np


class Network:
    def __init__(self, env):
        """Initializing the values"""
        self.pvariance = 0.1    # variance of initial parameters
        self.nhiddens = 5       # number of internal neurons

        self.ninputs = env.observation_space.shape[0]
        
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n
        # Initialize the training parameters randomly by using Gaussian distribution with zero mean and 0.1 variance
        self.W1 = np.random.randn(self.nhiddens, self.ninputs) * self.pvariance
        self.W2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance
        # Initilize biases to 0.0
        self.b1 = np.zeros(shape=(self.nhiddens, 1))
        self.b2 = np.zeros(shape=(self.noutputs, 1))
        

    def update(self, observation):
        '''Update the action'''
        observation.resize(self.ninputs,1)
        Z1 = np.dot(self.W1, observation) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.tanh(Z2)
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        return action


    def evaluate(self, nepisodes):
        '''Evaluate the model'''
        for e in range(nepisodes):
            observation = env.reset()
            total_reward = []
            for t in range(100):
                env.render()
                action = Network.update(self, observation)
                observation, reward, done, info = env.step(action)
                total_reward.append(reward)        
                if done:
                    print("Episode {} finished after {} timesteps".format(e+1, t+1))
                    break
            print('fitness = ', sum(total_reward))
          
        env.close()

# Evaluate 100 robots with randomly different connection weights
# using the CartPole and another classic control problem
for robot in range(100):
    print("CartPole Problem")
    print("Robot {}".format(robot+1))
    env = gym.make('CartPole-v1')   
    network = Network(env)
    network.evaluate(10)
    print("Done")

# print('\n#######')
# print("Acrobot Problem")
# for robot in range(100):
#     print("Robot {}".format(robot+1))
#     env2 = gym.make('Acrobot-v1')   
#     network2 = Network(env)
#     network2.evaluate(10)
#     print("Done")