# Image segmentation using MRF model
from PIL import Image
import numpy
from pylab import *
from scipy.cluster.vq import *
from scipy.signal import *
import cv2
import scipy

seg_avg = [0] *2
total_cost = 1
def main():
	# Read in image
	global total_cost
	img=Image.open('zuko.png')
	img=numpy.array(img)

	obs = init_config(img)
	while(total_cost > 0):
		#print (obs)
		#print(obs[540,960])
		obs = MRF(img,obs)
		#print(total_cost)
	scipy.misc.imsave('out.png',obs*20)


def assign_label(img,i,j):
	(M,N)=img.shape[0:2]
	row_mid = M/2
	col_mid = N/2
	sin_pot = abs((i-row_mid)/float(M)) + abs((j - col_mid)/float(N))
	return sin_pot

def init_config(img):
	threshold = 128
	(M,N)=img.shape[0:2]
	img_seg = numpy.zeros(shape=(M,N))
	for i in range(M):
		for j in range(N):
			if(assign_label(img,i,j) < 0.5):
				img_seg[i,j] = 1
			else:
				img_seg[i,j] = 0
	return img_seg

def MRF(img,obs):
	global total_cost
	total_cost = 0
	(M,N)=obs.shape[0:2]
	global seg_avg
	seg_avg = segment_avg(img,obs)
	for i in range(M):
		for j in range(N):
			# Find segmentation level which has min energy (highest posterior)
			cost=[energy(img,obs, k, i, j) for k in range(2)]
			total_cost += min(cost)
			#print (total_cost)
			obs[i,j]=cost.index(min(cost))
	print(total_cost)
	return obs


def energy(img,obs,label,i,j):
	beta=0.5
	singleton = log(singleton_potential(img,obs,label,i,j))
	doubleton = doubleton_potential(obs,label,i,j)
	return beta* doubleton + singleton*1.15

def delta(a,b):
	if (a==b):
		return -1
	else:
		return 1

def segment_avg(img,obs):
	global seg_avg
	(M,N)=obs.shape[0:2]
	count_0 = 0
	count_1 = 0
	for i in range(M):
		for j in range(N):
			if(obs[i,j] == 0):
				seg_avg[0] += img[i,j]
				count_0 += 1 
			else:
				seg_avg[1] += img[i,j]
				count_1 += 1 
	seg_avg[0] /= count_0
	seg_avg[1] /= count_1
	return seg_avg

def singleton_potential(img,obs,label,i,j):
	global seg_avg
	return abs(img[i,j] - seg_avg[label])
	

def doubleton_potential(obs,label,i,j):
	(M,N)=obs.shape[0:2]

	#find correct neighbors
	if (i==0 and j==0):
		neighbor=[(0,1), (1,0)]
	elif i==0 and j==N-1:
		neighbor=[(0,N-2), (1,N-1)]
	elif i==M-1 and j==0:
		neighbor=[(M-1,1), (M-2,0)]
	elif i==M-1 and j==N-1:
		neighbor=[(M-1,N-2), (M-2,N-1)]
	elif i==0:
		neighbor=[(0,j-1), (0,j+1), (1,j)]
	elif i==M-1:
		neighbor=[(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j==0:
		neighbor=[(i-1,0), (i+1,0), (i,1)]
	elif j==N-1:
		neighbor=[(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:
		neighbor=[(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
	
	return sum(delta(label,obs[i]) for i in neighbor)

if __name__=="__main__":
	main()	