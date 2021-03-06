
import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from mpl_toolkits.mplot3d import Axes3D


#SGDRegressor defaults:
#loss='squared_loss', penatly='12', alpha='0.0001'
#l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
#verbose=0, epsilone=0.1, random_state=None, learning_rate='invscaling'
#eta0=0.01, power_t=0.25, warm_starter=False, average=False


#D is the dimention
class SGDRegressor:

    def __init__(self,D):
        self.w=np.random.random(D)/np.sqrt(D)       #initialize weights with random normalized values
        self.lr=10e-2                               #Learning rate

    def partial_fit(self,X,Y):
        self.w+=self.lr*(Y-X.dot(self.w)).dot(X)    #update weights based on stochastic Gradient Descent

    def predict(self,X):
        return X.dot(self.w)





class FeatureTransformer:

    def __init__(self,env):

        # observation_examples=np.array([env.observation_space.sample() for x in range(10000)])

        observation_examples=np.random.random((2000,4))*2-2
        scaler=StandardScaler()             #An standardScalar instance

        #standardize data = mean equal to zero and variant equal to 1
        scaler.fit(observation_examples)    #Calculate mean and variant (will be used to standardize data)

        #state vector goes through RBF kernel to featurize
        #We use different variances to cover all the range (different part of it)


        #feature union is a pipline consisting of transformers in parallel (pipeline is usally in series)
        #here it has 4 RBFs in parallel
        #gamma is the variance of gaussian
        #n_components is the number of exemplars

        featurizer=FeatureUnion([
            ("rbf1",RBFSampler(gamma=0.05,n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000)),
        ])

        feature_examples=featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions=feature_examples.shape[1]

        self.scaler=scaler
        self.featurizer=featurizer



    def transform(self,s):
        s_scaled=self.scaler.transform(s)       #scale state s

        #return transformed state
        return self.featurizer.transform(s_scaled)


class Model:

    def __init__(self,env,feature_transformer):

        self.env=env
        self.models=[]
        self.feature_transformer=feature_transformer

        #create a Q model for each action (using SGDRegression)
        for i in range(env.action_space.n):
            model=SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self,s):
        s_feature=self.feature_transformer.transform(np.atleast_2d(s))

        return np.array([m.predict(s_feature)[0] for m in self.models])


    def update(self,s,a,G):
        s_feature=self.feature_transformer.transform(np.atleast_2d(s))
        #[G] to make it 2D
        self.models[a].partial_fit(s_feature,[G])


    def greedy_policy(self,s,eps):

        if np.random.random()<eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_episode(env,model,eps,gamma):
    s=env.reset()
    done=False
    totalreward=0
    t=0

    while not done and t<2000:
        action=model.greedy_policy(s,eps)
        s_previous=s
        s,reward,done,info=env.step(action)

        totalreward+=reward

        if done and t<199:
            reward=-200

        q_next=model.predict(s)
        assert(len(q_next.shape)==1)
        #update the model
        G=reward+gamma*np.max(q_next)
        # print("G:",model.predict(s)[0])
        model.update(s_previous,action,G)

        t+=1
        # print("s:",s)

    return totalreward






def plot_running_average(totalrewards):
    N=len(totalrewards)
    running_ave=np.empty(N)

    for t in range(N):
        running_ave[t]=totalrewards[max(0,t-100):(t+1)].mean()
    plt.plot(running_ave)
    plt.title("Running Average")
    plt.show()


def main():
    env=gym.make('CartPole-v0')
    ft=FeatureTransformer(env)
    model=Model(env,ft)

    gamma=0.99

    if 'monitor' in sys.argv:

        filename=os.path.basename(__file__).split('.')[0]
        monitor_dir='./'+filename+'_'+str(datetime.now())
        env=wrappers.Monitor(env,monitor_dir)

    N = 500

    totalrewards = np.empty(N)
    costs=np.empty(N)

    for n in range(N):
        eps = 0.1/np.sqrt(n+1)
        totalreward = play_episode(env, model, eps, gamma)
        totalrewards[n] = totalreward

        if n % 100==0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
            print("avg reward for last 100 episodes:", totalrewards[max(0, n - 100):(n + 1)].mean())


    print("avg reward for last 100 episodes:", 100*totalrewards[-100:].mean())

    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    plot_running_average(totalrewards)


if __name__=='__main__':
    main()