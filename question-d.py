import numpy as np
import random
import matplotlib.pyplot as plt
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
    
def generator_x (x_training, x_testing, p_st, p_end, s_trn, s_tst):
    for i in range (p_st+1, p_end+1) :
       x_training = np.hstack(((x_training[:,-2]**i).reshape(s_trn,1) ,x_training))
       x_testing = np.hstack(((x_testing[:,-2]**i).reshape(s_tst,1) ,x_testing))
    return x_training , x_testing

size = 60
x = np.linspace(-3, 3, size)
y = np.zeros((size,1))

mu, sigma = 0, 1 # mean and standard deviation
noise = np.random.normal(mu, sigma, size) #Create Gaussian Noise

y = 2*x + noise

x = np.reshape( np.append( x, np.ones(size)), (2, size)).T
x_training, x_testing, y_training, y_test = train_test_split(x, y, test_size=0.25)

#### from function 
wlin, error_training, error_testing = calculate_error_L1 (x_training, y_training, x_testing, y_test, x_training.shape[0], x_testing.shape[0])

#### use to plot the fucntion
x_line = np.linspace (-3, 3 ,2000)
y_line = x_line * wlin[0] + wlin[1]

fig, ax = plt.subplots()

print ("\nQuestion d m = 60:" )

x_training_degree14 ,x_testing_degree14 = generator_x (x_training, x_testing, 1 , 14, x_training.shape[0], x_testing.shape[0])
wlin_degree14, error_training_degree14, error_testing_degree14 = calculate_error_L1 (x_training_degree14, y_training, x_testing_degree14, y_test, x_training.shape[0], x_testing.shape[0])
print ("\n--Degree 5--")
print ("Training error: " , error_training_degree14)
print ("Leave One out: " ,leave_one_out(x_training_degree14, y_training))
print ("Five Fold Error: " ,five_fold(x_training_degree14, y_training))
print ("Testing error: " , error_training_degree14)

y_degree_m1=0
for i in range (0, 15):
    y_degree_m1 += x_line**(14-i)* wlin_degree14 [i]

size = 160
x = np.linspace(-3, 3, size)
y = np.zeros((size,1))

mu, sigma = 0, 1 # mean and standard deviation
noise = np.random.normal(mu, sigma, size) #Create Gaussian Noise

y = 2*x + noise

x = np.reshape( np.append( x, np.ones(size)), (2, size)).T
x_training, x_testing, y_training, y_test = train_test_split(x, y, test_size=0.25)

x_training_degree14 ,x_testing_degree14 = generator_x (x_training, x_testing, 1 , 14, x_training.shape[0], x_testing.shape[0])
wlin_degree14, error_training_degree14, error_testing_degree14 = calculate_error_L1 (x_training_degree14, y_training, x_testing_degree14, y_test, x_training.shape[0], x_testing.shape[0])
print ("\n--Degree 5--")
print ("Training error: " , error_training_degree14)
print ("Leave One out: " ,leave_one_out(x_training_degree14, y_training))
print ("Five Fold Error: " ,five_fold(x_training_degree14, y_training))
print ("Testing error: " , error_training_degree14)

y_degree_m2=0
for i in range (0, 15):
    y_degree_m2 += x_line**(14-i)* wlin_degree14 [i]


size = 320
x = np.linspace(-3, 3, size)
y = np.zeros((size,1))

mu, sigma = 0, 1 # mean and standard deviation
noise = np.random.normal(mu, sigma, size) #Create Gaussian Noise

y = 2*x + noise

x = np.reshape( np.append( x, np.ones(size)), (2, size)).T
x_training, x_testing, y_training, y_test = train_test_split(x, y, test_size=0.25)

x_training_degree14 ,x_testing_degree14 = generator_x (x_training, x_testing, 1 , 14, x_training.shape[0], x_testing.shape[0])
wlin_degree14, error_training_degree14, error_testing_degree14 = calculate_error_L1 (x_training_degree14, y_training, x_testing_degree14, y_test, x_training.shape[0], x_testing.shape[0])
print ("\n--Degree 5--")
print ("Training error: " , error_training_degree14)
print ("Leave One out: " ,leave_one_out(x_training_degree14, y_training))
print ("Five Fold Error: " ,five_fold(x_training_degree14, y_training))
print ("Testing error: " , error_training_degree14)

y_degree_m3=0
for i in range (0, 15):
    y_degree_m3 += x_line**(14-i)* wlin_degree14 [i]

ax.plot (x_line, y_degree_m1, 'r', label='m = 60' )
ax.plot (x_line, y_degree_m2, 'g', label='m = 160' )
ax.plot (x_line, y_degree_m3, 'b', label='m = 320' )

legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')

plt.show()
