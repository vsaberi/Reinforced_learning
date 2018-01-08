
import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


#SGDRegressor defaults:
#loss='squared_loss', penatly='12', alpha='0.0001'
#l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
#verbose=0, epsilone=0.1, random_state=None, learning_rate='invscaling'
#eta0=0.01, power_t=0.25, warm_starter=False, average=False



class FeatureTransformer:

    def __init__(self,env):

        # observation_examples=np.array([env.observation_space.sample() for x in range(10000)])

        observation_examples=np.random.random((2000,4))*2-2
        self.scaler=StandardScaler()             #An standardScalar instance

        #standardize data = mean equal to zero and variant equal to 1
        self.scaler.fit(observation_examples)    #Calculate mean and variant (will be used to standardize data)



#D is the dimention



class NN:
    def __init__(self,D):

        n=2
        self.w1=tf.Variable(tf.random_normal(shape=(D,n)),name='w1')
        self.b1=tf.Variable(tf.random_normal([n]),name='b1')

        self.w2=tf.Variable(tf.random_normal(shape=(n,1)),name='w2')
        self.b2=tf.Variable(tf.random_normal([1]),name='b2')


        self.X=tf.placeholder(tf.float32,shape=(None,D),name='X')
        self.Y=tf.placeholder(tf.float32,shape=(None,),name='Y')


        hidden_out=tf.add(tf.matmul(self.X,self.w1),self.b1)
        hidden_out=tf.nn.sigmoid(hidden_out)

        Y_hat=tf.add(tf.matmul(hidden_out,self.w2),self.b2)


        delta=self.Y-Y_hat

        cost=tf.reduce_sum(tf.matmul(delta,delta))

        self.train_op=tf.train.AdamOptimizer(learning_rate=10e-2).minimize(cost)

        self.predict_op=Y_hat


        init=tf.global_variables_initializer()
        self.session=tf.InteractiveSession()
        self.session.run(init)



    def partial_fit(self,X,Y):
        self.session.run(self.train_op,feed_dict={self.X:X,self.Y:Y})

    def predict(self,X):
        return self.session.run(self.predict_op,feed_dict={self.X:X})



class Model:

    def __init__(self,env):

        self.scaler=FeatureTransformer(env).scaler
        self.env=env
        self.models=[]

        #create a Q model for each action (using SGDRegression)
        for i in range(env.action_space.n):
            model=NN(env.observation_space.shape[0])
            self.models.append(model)

    def predict(self,s):
        s=self.scaler.transform(np.atleast_2d(s))
        return np.array([m.predict(s)[0] for m in self.models])


    def update(self,s,a,G):
        s=self.scaler.transform(np.atleast_2d(s))

        self.models[a].partial_fit(s,[G])


    def greedy_policy(self,s,eps):
        s=self.scaler.transform(np.atleast_2d(s))

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
        # assert(len(q_next.shape)==1)
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
    model=Model(env)

    gamma=0.99

    if 'monitor' in sys.argv:

        filename=os.path.basename(__file__).split('.')[0]
        monitor_dir='./'+filename+'_'+str(datetime.now())
        env=wrappers.Monitor(env,monitor_dir)

    N = 5000

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