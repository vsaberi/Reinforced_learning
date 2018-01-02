#this  code tries to control the inverted pendulum based on random weights (search randomly for optimal weights)
#where the action is defined based on sign of s and w inner product
# s is the state vector and w is the weight vector


import gym
import numpy as np
import matplotlib.pyplot as plt

#this function gets states and a weight w
# it returns action 1 or 0 based on s.w>0 or s.w<0

def get_action(s,w):
    return 1 if s.dot(w)>0 else 0


#This function plays one episode based on random weight
#It logs the length of each episode
#better random weights would result in longer episodes!

def play_one_episode(env,w):

    s_observation=env.reset()         #reset environment (initial state)
    done=False
    t=0

    while not done and t<1000:
        #env.render()
        t+=1
        action=get_action(s_observation,w)
        s_observation,reward,done,info=env.step(action)
        if done:
            break

    return t


#this function gets weight w and plays N episodes and returns the average episode length

def play_multiple_episodes(env,N,w):

    episode_lengths=np.empty(N)

    for i in range(N):
        episode_lengths[i]=play_one_episode(env,w)

    avg_length=episode_lengths.mean()
    print("average length:",avg_length)
    return avg_length


#this function generates random weights and runs 100 episodes for each weight
#it returns the average length of 100 episodes and the best weights

def random_search(env):
    episode_length=[]
    best_length=0
    w=None

    for t in range(100):
        w=np.random.random(4)*2-1
        avg_length=play_multiple_episodes(env,100,w)
        episode_length.append(avg_length)


    if avg_length>best_length:
        w_best=w
        best_length=avg_length
    return episode_length,w_best





if __name__=='__main__':
    env=gym.make('CartPole-v0')         #inverted pendulum environment
    avg_episode_length, w_best=random_search(env)
    plt.plot(avg_episode_length)
    plt.show()

    print("final run with final weights")
    play_multiple_episodes(env,100,w_best)
