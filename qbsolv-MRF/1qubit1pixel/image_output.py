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

def main():
	file_name = sys.argv[1]
	img_name, ext = file_name.split(".")
	seg_file_name = img_name + "_segmentation.qbout"
	img=Image.open(file_name)
	img=numpy.array(img)
	(M,N) = img.shape[0:2]

	counter = 0
	array =[]
	with open(seg_file_name, "r") as ins:
		for line in ins:
			if(counter == 1):
				array = (line)
			counter += 1
	
	counter = 1
	out = []
	for i in array:
		if(counter > (N*M)):
			continue
		if i == '1':
			out.append(1)
		else:
			out.append(0)
		counter += 1

	out= numpy.array(out)
	out = out.reshape(M,N)
	
	img_name, ext = file_name.split(".")
	output_file_name = str(img_name) + "_out." + str(ext)
	scipy.misc.imsave(output_file_name,out)

if __name__=="__main__":
	main()	


