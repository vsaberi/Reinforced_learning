import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

#use some code already written
import Mountain_car_q_learning
from Mountain_car_q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_average

class LambdaRegressor:

    def __init__(self,D):
        self.w = np.random.random(D) / np.sqrt(D)

    def partial_fit(self,X,Y,eligibility,lr=10e-3):
        self.w+=lr*(Y-X.dot(self.w))*eligibility

    def predict(self,X):
        X=np.array(X)
        return X.dot(self.w)


#holds a regressor model for each action
class Model:

    def __init__(self,env,feature_transformer):
        self.env=env
        self.models=[]
        self.feature_transformer=feature_transformer


        D=2000
        self.eligibilities=np.zeros((env.action_space.n,D))
        for i in range(env.action_space.n):
            model=LambdaRegressor(D)
            self.models.append(model)

    def predict(self,s):
        s_transformed=self.feature_transformer.transform([s])
        assert(len(s_transformed.shape)==2)
        return np.array([m.predict(s_transformed)[0] for m in self.models])

    def update(self,s,a,G,gamma,lambda_):
        s_transformed=self.feature_transformer.transform([s])
        assert(len(s_transformed.shape)==2)
        self.eligibilities*=gamma*lambda_
        self.eligibilities += s_transformed[0]
        self.models[a].partial_fit(s_transformed[0],G,self.eligibilities[a])

    def greedy_policy(self,s,eps):
        if np.random.random()<eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_episode(env,model,eps,gamma,lambda_):
    s=env.reset()
    done=False
    totalrewards=0
    t=0


    while not done and t<200:

        action=model.greedy_policy(s,eps)
        s_previous=s
        s,reward,done,info=env.step(action)

        G=reward+gamma*np.max(model.predict(s)[0])
        model.update(s_previous,action,G,gamma,lambda_)

        totalrewards+=reward
        t+=1

    return totalrewards





if __name__=='__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)

    gamma = 0.99
    lambda_=0.7

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 1500

    totalrewards = np.empty(N)
    costs=np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97 ** (n / 10))
        # eps=0.01
        totalreward = play_episode(env, model, eps, gamma,lambda_)
        totalrewards[n] = totalreward

        print("episode:", n, "total reward:", totalreward, "eps:", eps)
        print("avg reward for last 100 episodes:", totalrewards[max(0, n - 100):(n + 1)].mean())

    print("avg reward for last 100 episodes:", 100 * totalrewards[-100:].mean())

    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    plot_running_average(totalrewards)

    plot_cost_to_go(env, model)





