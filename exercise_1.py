import gym

env = gym.make('CartPole-v1')

for i_episode in range(10):
    observation = env.reset()
    total_reward = []
    
    for t in range(100):
        env.render()
        print('Observation:', observation)
        action = env.action_space.sample()
        print('action:', action)
        observation, reward, done, info = env.step(action)
        total_reward.append(reward)        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print('fitness = ', sum(total_reward))
env.close()

print('Observation space high:\n', env.observation_space.high)
print('Observation space low:\n', env.observation_space.low)
print('Action space:', env.action_space)