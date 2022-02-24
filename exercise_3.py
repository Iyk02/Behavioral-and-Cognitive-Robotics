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
        

    def set_genotype(self, population):
        # Initialize the training parameters
        index1 = self.nhiddens * self.ninputs
        pop = population[:index1]
        self.W1 = pop.reshape(self.nhiddens, self.ninputs) * self.pvariance
        index2 = index1 + self.nhiddens
        index3 = index2 + self.nhiddens * self.noutputs
        pop = population[index2:index3]
        self.W2 = pop.reshape(self.noutputs, self.nhiddens) * self.pvariance
        self.b1 = np.zeros(shape=(self.nhiddens, 1))
        self.b2 = np.zeros(shape=(self.noutputs, 1))

    # Compute number of parameters
    def compute_nparameters(self):
        nparameters = (self.nhiddens * self.ninputs) + (self.noutputs * self.nhiddens) + self.nhiddens + self.noutputs

        return nparameters

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
        reward_per_episode = []
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
            reward_per_episode.append(sum(total_reward))
          
        env.close()
        print('Average fitness =', np.mean(reward_per_episode))
        return np.mean(reward_per_episode)


env = gym.make('CartPole-v1')   
network = Network(env)
popsize = 10
ngenerations = 100
episodes = 3
nparameters = network.compute_nparameters()
perturb_variance = 0.02
population = np.random.randn(popsize, nparameters) * 0.1

for g in range(ngenerations):
    print("Generation {}".format(g+1))
    fitness = []
    # evaluating the individuals
    for i in range(popsize):
        print("Population {}".format(i+1))
        network.set_genotype(population[i])
        fit = network.evaluate(episodes)
        fitness.append(fit)

    indexbest = np.argsort(fitness)
    
    # replacing the worst genotypes with perturb versions of the best genotypes
    for i in range(int(popsize/2)):
        population[indexbest[i]] = population[indexbest[i+5]] + np.random.randn(nparameters) * perturb_variance
