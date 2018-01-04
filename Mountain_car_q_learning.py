
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


#SGDRegressor defaults:
#loss='squared_loss', penatly='12', alpha='0.0001'
#l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
#verbose=0, epsilone=0.1, random_state=None, learning_rate='invscaling'
#eta0=0.01, power_t=0.25, warm_starter=False, average=False



class FeatureTransformer:

    def __init__(self,env):

        observation_examples=np.array([env.observation_space.sample() for x in range(10000)])
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
            ("rbf1",RBFSampler(gamma=5.0,n_components=500)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=500)),
        ])

        featurizer.fit(scaler.transform(observation_examples))


        self.scaler=scaler
        self.featurizer=featurizer



    def transform(self,s):
        s_scaled=self.scaler.transform(s)       #scale state s

        #return transformed state
        return self.featurizer.transform(s_scaled)


class Model:

    def __init__(self,env,feature_transformer,learning_rate):

        self.env=env
        self.models=[]
        self.feature_transformer=feature_transformer

        #create a Q model for each action (using SGDRegression)
        for i in range(env.action_space.n):
            model=SGDRegressor(learning_rate=learning_rate)

            #the initial target is zero
            #the target is inside [] (since the default is 2D)
            model.partial_fit(feature_transformer.transform([env.reset()]),[0])
            self.models.append(model)

    def predict(self,s):

        s_feature=self.feature_transformer.transform([s])

        #make sure s is 2D (s should be inside [] to make it 2D)
        assert(len(s_feature.shape)==2)
        return np.array([m.predict(s_feature)[0] for m in self.models])


    def update(self,s,a,G):
        s_feature=self.feature_transformer([s])
        assert(len(s_feature.shape)==2)

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

    while not done and iter<10000:
        action=model.greedy_policy(s,eps)
        s_previous=s
        s,reward,done,info=env.step(action)

        #update the model
        G=reward+gamma*np.max(model.predict(s)[0])
        model.update(s_previous,action,G)

        totalreward+=reward
        t+=1

        return totalreward


    

