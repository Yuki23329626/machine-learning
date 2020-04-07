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
    
def generator_x (x_training, x_test, p_st, p_end, s_trn, s_tst):
    for i in range (p_st+1, p_end+1) :
       x_training = np.hstack(((x_training[:,-2]**i).reshape(s_trn,1) ,x_training))
       x_test = np.hstack(((x_test[:,-2]**i).reshape(s_tst,1) ,x_test))
    return x_training , x_test

fig, ax = plt.subplots()

x = np.linspace(0, 1, 20)
y = np.zeros((20,1))

mu, sigma = 0, 0.04 # mean and standard deviation
noise = np.random.normal(mu, sigma, 20) #Create Gaussian Noise

y = np.sin(2*np.pi*x) + noise
ax.scatter (x, y, c='c')

x = np.reshape( np.append( x, np.ones(20)), (2, 20)).T
x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.25)

#### from function 
wlin, error_training, error_testing = calculate_error_L1 (x_training, y_training, x_test, y_test, 15, 5)

#### use to plot the fucntion
x_line = np.linspace (0, 1 ,1000)
y_line = x_line * wlin[0] + wlin[1]


print ("\nQuestion c:" )

x_training_degree5 ,x_testing_degree5 = generator_x (x_training, x_test, 1 , 5, 15, 5)
wlin_degree5, error_training_degree5, error_testing_degree5 = calculate_error_L1 (x_training_degree5, y_training, x_testing_degree5, y_test, 15, 5)
print ("\n--Degree 5--")
print ("Training error: " , error_training_degree5)
print ("Leave One out: " ,leave_one_out(x_training_degree5, y_training))
print ("Five Fold Error: " ,five_fold(x_training_degree5, y_training))
print ("Testing error: " , error_training_degree5)

y_degree5=0
for i in range (0, 6):
    y_degree5 += x_line**(5-i)* wlin_degree5 [i]

x10_train, x10_test = generator_x (x_training_degree5, x_testing_degree5 , 5, 10, 15, 5)
x10_wlin, x10_train_error, x10_test_error = calculate_error_L1 (x10_train, y_training, x10_test, y_test, 15, 5)
print ("\n--Degree 10--")
print ("Training error: " , x10_train_error)
print ("Leave One out: " ,leave_one_out(x10_train, y_training))
print ("Five Fold Error: " ,five_fold(x10_train, y_training))
print ("Testing error: " , x10_test_error)

y_degree10=0
for i in range (0, 11):
    y_degree10 += x_line**(10-i)* x10_wlin [i]

x14_train, x14_test = generator_x (x10_train, x10_test , 10, 14, 15, 5)
x14_wlin, x14_train_error, x14_test_error = calculate_error_L1 (x14_train, y_training, x14_test, y_test, 15, 5)
print ("\n--Degree 14--")
print ("Training error: " , x14_train_error)
print ("Leave One out: " ,leave_one_out(x14_train, y_training))
print ("Five Fold Error: " ,five_fold(x14_train, y_training))
print ("Testing error: " , x14_test_error)

y_degree14=0
for i in range (0, 15):
    y_degree14 += x_line**(14-i)* x14_wlin [i]



ax.plot (x_line, y_degree5 , 'r', label='degree 5')
ax.plot (x_line, y_degree10, 'g', label='degree 10' )
ax.plot (x_line, y_degree14, 'b', label='degree 14' )

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()
