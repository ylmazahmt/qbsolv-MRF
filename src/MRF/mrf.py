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
import sys,os
from subprocess import call

def unary_potential(img,label,i,j):
	if(label == 0):
		return (img[i,j]/255.0) - 1
	else:
		return (0 - (img[i,j]/255.0))
	

def find_neighbors(img,i,j):
	(M,N)=img.shape[0:2]
	#find correct neighbors
	
	#right-bottom corner(no post neighbors available for this pixel)
	if i==M-1 and j==N-1:
		neighbor=[]
	#bottom line
	elif i==M-1:
		neighbor=[(M-1,j+1)]
	#right line
	elif j==N-1:
		neighbor=[(i+1,N-1)]
	#any other pixel
	else:
		neighbor=[(i,j+1), (i+1,j+1), (i+1,j)]

	return neighbor

def f_1(img,i,j,k):
	k_1 = 1
	cost = (k_1 * (abs(img[i,j] - img[k])/255.0))
	return cost

def f_2(img,i,j,k):
	k_2 = 0.025
	cost = k_2 * ((1 - (abs(img[i,j] - img[k])/255.0)))
	return cost


def energy(img,seg,label,i,j):
	singleton = singleton_potential(img,label,i,j)
	doubleton = doubleton_potential(img,seg,label,i,j)
	return  singleton + 0.025*doubleton

def resulting_energy(img,seg):
	(M,N)=seg.shape[0:2]
	cost = 0
	for i in range(M):
		for j in range(N):
			cost += energy(img,seg,seg[i,j],i,j)
	print("Resulting energy is: ", cost)
	return cost

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
