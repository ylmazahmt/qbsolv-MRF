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
import sys,inspect,os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
args = cmd_folder.split('/')
src_folder = '/'.join(args[:-1])
sys.path.insert(0, (src_folder))

from MRF.mrf import *

def main():
	img_source_path = sys.argv[1]
	args = img_source_path.split("/")
	file_name = args[-1]
	img_name, ext = file_name.split(".")
	seg_file_name = img_name + "_segmentation.qbout"
	rel_path = "result/" +img_name + "/"
	file_path = rel_path + seg_file_name
	img=Image.open(img_source_path)
	img=numpy.array(img)
	(M,N) = img.shape[0:2]

	counter = 0
	# Read in image
	array =[]
	with open(file_path, "r") as ins:
		for line in ins:
			if(counter == 1):
				array = (line)
			counter += 1
	
	counter = 1
	out = []
	for i in array:
		mode = counter
		if(counter > (N*M*2)):
			continue
		if(mode%2 == 1):
			if i == '1':
				out.append(1)
			else:
				out.append(0)
		counter += 1

	out= numpy.array(out)
	out = out.reshape(M,N)

	resulting_energy(img,out)

	output_file_name = str(img_name) + "_out." + str(ext)
	file_path = rel_path + output_file_name
	scipy.misc.imsave(file_path,out)
	
if __name__=="__main__":
	main()	


