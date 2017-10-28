import numpy as np
import k_nearest_neighbor as knn
import sys

x = np.array([[1,2],[3,4]])
y = np.array([0,1])
test = np.array([[1,2],[3,4]])

if __name__ == '__main__':
	if len(sys.argv) == 2:
		choice = sys.argv[1]
		
		if choice == '2':
			kk = knn.KNearestNeighbor()
			kk.train(x,y)
			dist = kk.compute_distances_two_loops(test)
			print dist
			print kk.predict_labels(dist)
			
		elif choice == '1':
			kk = knn.KNearestNeighbor()
			kk.train(x,y)
			dist = kk.compute_distances_one_loop(test)
			print dist
			print kk.predict_labels(dist)
			
		elif choice == '0':
			kk = knn.KNearestNeighbor()
			kk.train(x,y)
			dist = kk.compute_distances_no_loops(test)
			print dist
			print kk.predict_labels(dist)
		
		else: 
			print 'error'
			
	else:
		print 'error'