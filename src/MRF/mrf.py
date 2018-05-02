# Image segmentation using MRF model
from PIL import Image
import numpy as np
from pylab import *
from scipy.cluster.vq import *
from scipy.signal import *
import cv2
import scipy
import math
from random import randint
import sys,os
from subprocess import call

binsForHistogramIntersection = 255

def histogram_intersection(segment_1, segment_2):
	# print("segment 1:",segment_1)
	# print("segment 2:",segment_2)
	hist_1, _ = np.histogram(segment_1, bins=binsForHistogramIntersection, range=[0, 255])
	hist_2, _ = np.histogram(segment_2, bins=binsForHistogramIntersection, range=[0, 255])
	# print("hist 1:",hist_1)
	# print("hist 2:",hist_2)
	minima = np.minimum(hist_1, hist_2)
	# print("minima", minima)

	intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
	# print("intersection", intersection)
	return intersection

		
def unary_potential(superpixels,label,i,foregroundModel,backgroundModel):
	if(label == 0):
		# print("Intersection with background: ",i)
		return 0 - histogram_intersection(superpixels[i],backgroundModel)
	else:
		# print("Intersection with foreground: ",i)
		return 0 - histogram_intersection(superpixels[i],foregroundModel)

def doubleton_potential(superpixels,label,neighborLabel,i,neighbor):
	if label ==  neighborLabel:
		# print("Intersection of: ",i," with ", neighbor)
		doubleton = 0 - histogram_intersection(superpixels[i],superpixels[neighbor])
	else:
		# print("Intersection of: ",i," with ", neighbor)
		doubleton = histogram_intersection(superpixels[i],superpixels[neighbor]) - 1
	return doubleton


# Required only for ICM algorithm
def energy(superpixels,seg,label,i,segNeighbors,foregroundModel,backgroundModel):
	singleton = unary_potential(superpixels,label,i,foregroundModel,backgroundModel)
	doubleton = 0
	for neighbor in segNeighbors[i]:
		doubleton += doubleton_potential(superpixels,label,seg[neighbor],i,neighbor)
	# print("singleton: ",singleton,"doubleton: ",doubleton)
	return  singleton + 0.01*doubleton

