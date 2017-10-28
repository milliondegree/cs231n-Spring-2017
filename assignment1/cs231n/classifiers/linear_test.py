import numpy as np
import linear_classifier as lc

s = lc.Softmax()
v = lc.LinearSVM()
x = np.array([[1,2],[2,4],[3,6],[4,8],[5,10],[6,12],[2,1],[4,2],[6,3],[8,4],[10,5],[12,6]])
#y = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])
y = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
t = np.array([[7,14],[20,10],[3,2],[4,5]])

s.train(x,y)
print s.predict(t)
print ' '
v.train(x,y)
print v.predict(t)


