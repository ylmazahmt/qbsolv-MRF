# Image segmentation using MRF model
from PIL import Image
import numpy
from pylab import *
from scipy.cluster.vq import *
from scipy.signal import *
import cv2
import scipy
import math
from random import randint


total_cost = 1000000
threshold = 2000
def main():
	# Read in image
	global total_cost
	old_total_cost = 0
	img=Image.open('zuko.png')
	img=numpy.array(img)
	(M,N)=img.shape[0:2]
	print(M,N)
	seg = init_config(img)
	while(abs(total_cost - old_total_cost) > threshold):
		#print (seg)
		#print(seg[540,960])
		old_total_cost = total_cost
		seg = ICM(img,seg)
		print(total_cost)
	scipy.misc.imsave('out.png',seg*255)


def init_config(img):
	(M,N)=img.shape[0:2]
	img_seg = numpy.zeros(shape=(M,N))
	for i in range(M):
		for j in range(N):
			img_seg[i,j] = randint(0, 1)
	return img_seg

def ICM(img,seg):
	global total_cost
	total_cost = 0
	(M,N)=seg.shape[0:2]
	for i in range(M):
		for j in range(N):
			# Find segmentation level which has min energy (highest posterior)
			cost=[energy(img,seg, k, i, j) for k in range(2)]
			total_cost += min(cost)
			#print (total_cost)
			seg[i,j]=cost.index(min(cost))
	return seg


def energy(img,seg,label,i,j):
	singleton = singleton_potential(img,label,i,j)
	doubleton = doubleton_potential(img,seg,label,i,j)
	return  singleton + 0*doubleton

def delta(a,b):
	if (a==b):
		return 0
	else:
		return 1

def singleton_potential(img,label,i,j):
	return abs((img[i,j]/255.0) - label)


def doubleton_potential(img,seg,label,i,j):
	(M,N)=seg.shape[0:2]
	cost = 0
	k_1 = 1
	k_2 = 0
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

	for k in neighbor:
		if (img[i,j] - img[k]) == 0:
				continue
		if delta(label,seg[k]) == 0:
			#print(log(2.71))
			cost += k_1 * (abs(img[i,j] - img[k])/255.0)
		else:
			#print(k_2 * (1 - abs(img[i,j] - img[k])/255.0))
			cost += k_2 * (1 - (abs(img[i,j] - img[k])/255.0))


	
	return cost

if __name__=="__main__":
	main()	
