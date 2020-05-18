import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

def linear_regression(x,y):
  x = np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
  y = y[:,np.newaxis]
  w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
  return w_lin

def ridge_regression(x,y):
  x = np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
  y = y[:,np.newaxis]
  w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)+LAMDA*np.identity(15)),x.T),y)
  return w_lin2

def cal_error ( x_train, y_train, x_val, y_val, trn_s, val_s):
    wln = np.linalg.inv(x_train.T.dot( x_train)).dot(x_train.T.dot(y_train))
    trn_error = abs((x_train.dot(wln) - y_train).sum()/trn_s)
    val_error = abs((x_val.dot(wln) - y_val).sum()/val_s)
    return  wln, trn_error, val_error
    #return abs(x_val.dot(wln) - y_val).sum()/size

def leave_one_out(x ,y ):
    loo = LeaveOneOut()
    leave_one_out = 0
    for train_index, test_index in loo.split(x) : 
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index] 
        leave_one_out += cal_error(X_train, Y_train, X_test, Y_test, 14, 1)[2]
    return(leave_one_out/15)

def five_fold(x, y):
    #### five fold error
    kf = KFold (n_splits = 5)
    five_fold_error = 0
    for train_index , test_index in kf.split( x_train ):
        #print ("Train: " ,train_index , "Test: ", test_index)
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        five_fold_error += cal_error(X_train, Y_train, X_test, Y_test,12, 3)[2]
    return (five_fold_error/5)
    
def generator_x (x_train, x_test, p_st, p_end, s_trn, s_tst):
    for i in range (p_st+1, p_end+1) :
       x_train = np.hstack(((x_train[:,-2]**i).reshape(s_trn,1) ,x_train))
       x_test = np.hstack(((x_test[:,-2]**i).reshape(s_tst,1) ,x_test))
    return x_train , x_test


x = np.arange(-3, 3+6/19, 6/19)
y = np.zeros((20,1))

for i in range(20):
    #t=  random.random()
    y[i] = 2*x[i] + random.random()

x = np.reshape( np.append( x, np.ones(20)), (2, 20)).T
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#### from function 
wln, train_error, test_error = cal_error (x_train, y_train, x_test, y_test, 15, 5)
print("Training error  = " , train_error)
print ("Testing error  = " , test_error)

#### use to plot the fucntion
x_ln = np.linspace (-3, 3 ,1000)
y_ln = x_ln * wln[0] + wln[1]

plt.plot (x_ln, y_ln )
plt.scatter (x_train[:,0], y_train)

print ("Five Fold Error = " ,five_fold(x_train,y_train))
print ("Leave One out = " ,leave_one_out(x_train, y_train))

####Q2

x5_train ,x5_test = generator_x (x_train, x_test, 1 , 5, 15, 5)
x5_wln, x5_train_error, x5_test_error = cal_error (x5_train, y_train, x5_test, y_test, 15, 5)
print ("================================================================")
print ("5 D")
print("Training error  = " , x5_train_error)
print("Testing error  = " , x5_test_error)
print ("Five Fold Error = " ,five_fold(x5_train, y_train))
print ("Leave One out = " ,leave_one_out(x5_train, y_train))
y_5=0
for i in range (0, 6):
    y_5 += x_ln**(5-i)* x5_wln [i]
x10_train, x10_test = generator_x (x5_train, x5_test , 5, 10, 15, 5)
x10_wln, x10_train_error, x10_test_error = cal_error (x10_train, y_train, x10_test, y_test, 15, 5)
print ("================================================================")
print ("10 D")
print("Training error  = " , x10_train_error)
print("Testing error  = " , x10_test_error)
print ("Five Fold Error = " ,five_fold(x10_train, y_train))
print ("Leave One out = " ,leave_one_out(x10_train, y_train))
y_10=0
for i in range (0, 11):
    y_10 += x_ln**(10-i)* x10_wln [i]

x14_train, x14_test = generator_x (x10_train, x10_test , 10, 14, 15, 5)
x14_wln, x14_train_error, x14_test_error = cal_error (x14_train, y_train, x14_test, y_test, 15, 5)
print ("================================================================")
print ("14 D")
print("Training error  = " , x14_train_error)
print("Testing error  = " , x14_test_error)
print ("Five Fold Error = " ,five_fold(x14_train, y_train))
print ("Leave One out = " ,leave_one_out(x14_train, y_train))
y_14=0
for i in range (0, 15):
    y_14 += x_ln**(14-i)* x14_wln [i]

plt.plot (x_ln, y_5 )
plt.plot (x_ln, y_10 )
plt.plot (x_ln, y_14 )

