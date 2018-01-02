

import gym



env=gym.make('CartPole-v0')         #inverted pendulum environment

print(env.reset())         #reset env to random state

# These are the sates: [cart position, cart velocity, pole angle, pole tip velocity]

box=env.observation_space           #state space

print(box.low)          #min value of possible states
print(box.high)         #max value of possible states
print(box.sample())     #a sample of state space

actions=env.action_space        #action space
#
print(actions.n)   #number of actions in action space


#random step!
done=False
k=0
while not done:
    observation, reward, done, info =env.step(actions.sample())
    k=k+1
    print(observation)
    print(reward)
    print(done)
    print(info)
    print(k)