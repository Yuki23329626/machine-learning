import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

def linear_regression ( x_training, y_training, x_valid, y_valid, sum_training, sum_valid):
  wlin = np.linalg.inv(x_training.T.dot( x_training)).dot(x_training.T.dot(y_training))
  return  wlin

def ridge_regression ( x_training, y_training, x_valid, y_valid, sum_training, sum_valid, lamda):
  wlin = np.linalg.inv(x_training.T.dot( x_training)).dot(x_training.T.dot(y_training)+lamda*np.eye(x_training[0]))
  return  wlin

def calculate_error_L1 ( x_training, y_training, x_valid, y_valid, sum_training, sum_valid):
    wlin = linear_regression( x_training, y_training, x_valid, y_valid, sum_training, sum_valid)
    error_training = abs((x_training.dot(wlin) - y_training).sum()/sum_training)
    error_testing = abs((x_valid.dot(wlin) - y_valid).sum()/sum_valid)
    return  wlin, error_training, error_testing

def calculate_error_L2 ( x_training, y_training, x_valid, y_valid, sum_training, sum_valid):
    wlin = ridge_regression( x_training, y_training, x_valid, y_valid, sum_training, sum_valid)
    error_training = abs((x_training.dot(wlin) - y_training).sum()/sum_training)
    error_testing = abs((x_valid.dot(wlin) - y_valid).sum()/sum_valid)
    return  wlin, error_training, error_testing

def leave_one_out(x ,y ):
    leaveOneOut = LeaveOneOut()
    leave_one_out = 0
    for train_index, test_index in leaveOneOut.split(x) : 
        x_training, X_test = x[train_index], x[test_index]
        y_training, Y_test = y[train_index], y[test_index] 
        leave_one_out += calculate_error_L1(x_training, y_training, X_test, Y_test, 14, 1)[2]
    return(leave_one_out/15)

def five_fold(x, y):
    #### five fold error
    kf = KFold (n_splits = 5)
    five_fold_error = 0
    for index_training , index_testing in kf.split( x_training ):
        #print ("Train: " ,train_index , "Test: ", test_index)
        X_training, X_testing = x[index_training], x[index_testing]
        Y_training, Y_testing = y[index_training], y[index_testing]
        five_fold_error += calculate_error_L1(X_training, Y_training, X_testing, Y_testing,12, 3)[2]
    return (five_fold_error/5)
    
def generator_x (x_training, x_test, p_st, p_end, s_trn, s_tst):
    for i in range (p_st+1, p_end+1) :
       x_training = np.hstack(((x_training[:,-2]**i).reshape(s_trn,1) ,x_training))
       x_test = np.hstack(((x_test[:,-2]**i).reshape(s_tst,1) ,x_test))
    return x_training , x_test


x = np.linspace(-3, 3, 20)
y = np.zeros((20,1))

mu, sigma = 0, 1 # mean and standard deviation
noise = np.random.normal(mu, sigma, 20) #Create Gaussian Noise

y = 2*x + noise

x = np.reshape( np.append( x, np.ones(20)), (2, 20)).T
x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.25)

print ("Question a:\n" )

#### from function 
wlin, error_training, error_testing = calculate_error_L1 (x_training, y_training, x_test, y_test, 15, 5)
print("Training error: " , error_training)
print ("Testing error: " , error_testing)

#### use to plot the fucntion
x_line = np.linspace (-3, 3 ,1000)
y_line = x_line * wlin[0] + wlin[1]

fig, ax = plt.subplots()

ax.plot (x_line, y_line )
ax.scatter (x_training[:,0], y_training, c='k')

print ("Five Fold Error: " ,five_fold(x_training,y_training))
print ("Leave One out: " ,leave_one_out(x_training, y_training))

plt.show()
