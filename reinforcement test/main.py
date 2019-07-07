import gym
from Reinforcement.dnn import Agent



if __name__ == "__main__":
    
    # declare the environment
    env = gym.make("MountainCar-v0")
    
    # initial the q table in dictionary
    agent = Agent()

    # for i_episode in range(20000):
    t_ = 10000
    while True:
        
        observation, reward = [env.reset(), 0]

        for t in range(t_):
            env.render()
            
            # sent to agent
            act = agent._action(observation,reward)

            observation, reward, done, info = env.step(act)
            if done:
                print("Episode finished after {} timesteps.".format(t+1))
                break