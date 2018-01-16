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
        self.featuretransformer=feature_transformer


        D=feature_transformer.dimensions
        self.eligibilities=np.zeros((env.actions_space.n,D))
        for i in range(env.action_space.n):
            model=LambdaRegressor(D)
            self.models.append(model)

    def predict(self,s):
        s_transformed=


