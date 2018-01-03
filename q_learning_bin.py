
import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import os
import datetime
from gym import wrappers

#this function turns list of integers into an int
#build_state([1,2,3,4,5]) -> 12345

def build_state(features):
    return int("".joint(map(lambda feature: str(int(feature)),features)))


def to_bin(value,bins):
    return np.digitize(x=[value],bins=bins)[0]



class FeatureTransformer:

    def __init__(self):

        self.cart_position_bins=np.linspace(-2.4,2.4,9)
        self.cart_velocity_bins=np.linspace(-2,2,9)
        self.pole_angle_bins=np.linspace(-0.4,0.4,9)
        self.pole_velocity_bins=np.linspace(-3.5,3.5,9)

    # return the feature as an #### int (4 bins indices together)

    def transform(self,observation):

        cart_pos, cart_vel, pole_angle, pole_vel = observation

        return build_state([

            to_bin(cart_pos,self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins),
        ])


class Model:

    def __init__(self,env,feature_transformer):
        self.env=env
        self.feature_transformer=feature_transformer

        num_states=10**env.observation_space.shape[0]
        num_actions=env.action_space.n

        self.Q=np.random.uniform(low=-1,high=1,size=(num_states,num_actions))

    #returns Q for all actions
    def predict_Q(self,s):
        index=self.feature_transformer.transform(s)
        return self.Q[index]

    #given return G for state action pair (s,a) we update Q(s,a)
    def update_Q(self,a,s,G):
        index=self.feature_transformer.transform(s)
        alpha=10e-3                                     #learning rate
        self.Q[index,a]+=alpha*(G-self.Q[index,a])

    #we take random action with small probability epsilon
    #otherwise we take the action with highest predicted Q
    def greedy_policy(self,s,eps):
        if np.random.random()<eps:
            return self.env.action_space.sample()
        else:
            Q_s=self.predict_Q(s)       #Q_s for all actions
            return np.argmax(Q_s)


def play_episode(env,model,eps,gamma):

    s=env.reset()
    done=False
    total_reward=0
    t=0

    while not done and t<1000:
        a=model.greedy_policy(s,eps)
        s_previous=s
        s,r,done,info=env.step(a)

        total_reward+=r


        if done and t<199:
            r=-300


        #update Q
        Q_max=np.max(model.predict_Q(s))        #maximum Q possible from sate s (by taking action 0 or 1)
        G=r+gamma*Q_max
        model.update_Q(s_previous,a,G)


        t+=1

    return total_reward



def plot_running_avg(totalrewards):

    N=len(totalrewards)
    running_avg=np.empty(N)

    for t in range(N):
        running_avg[t]=totalrewards[max(0,t-100):(t+1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()



if __name__ == '__main__':

    env=gym.make('CartPole-v0')
    ft=FeatureTransformer()

    model=Model(env,ft)
    gamma=0.9

    if 'monitor' in sys.argv:

        filename=os.path.basename(__file__).split('.')[0]
        monitor_dir='./'+filename+'_'+str(datetime.now())
        env=wrappers.Monitor(env,monitor_dir)

    N=10000

    totalrewards=np.empty(N)

    for n in range(N):
        eps=1.0/np.sqrt(n+1)

        totalreward=play_episode(env,model,eps,gamma)
        totalrewards[n]=totalreward

        if n % 100 ==0:
            print("episode:",n,"total reward:",totalreward,"eps:",eps)
            print("total steps:",totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
