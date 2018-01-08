import numpy as np
import theano
import theano.tensor as T
import RBF_Cart_pole_q_learning



class SGDRegressor:

    def __init__(self,D):
        w=np.random.random(D)/np.sqrt(D)
        self.w=theano.shared(w)              #create shared variable from w (self.w is not symbolic and it has stored value in memory)
        self.lr=10e-2

        X=T.matrix('X')
        Y=T.vector('Y')

        Y_hat=X.dot(self.w)
        delta=Y-Y_hat
        cost=delta.dot(delta)           #cost=delta^2
        grad=T.grad(cost,self.w)        #gradient
        updates=[(self.w,self.w-self.lr*grad)]

        self.train_op=theano.function(
            inputs=[X,Y],
            updates=updates,
        )

        self.predict_op=theano.function(
            inputs=[X],
            outputs=Y_hat,
        )

    def partial_fit(self,X,Y):
        self.train_op(X,Y)

    def predict(self,X):
        return self.predict_op(X)



if __name__=="__main__":
    RBF_Cart_pole_q_learning.SGDRegressor=SGDRegressor          #replace SGDRegressor function with theano written one
    RBF_Cart_pole_q_learning.main()