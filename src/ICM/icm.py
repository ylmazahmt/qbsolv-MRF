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
import sys,os,inspect
from subprocess import call

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
args = cmd_folder.split('/')
src_folder = '/'.join(args[:-1])
sys.path.insert(0, (src_folder))

from MRF.mrf import *

total_cost = 1000000
threshold = 2000
def main():
	# Read in image
	global total_cost
	old_total_cost = 0

	file_name = sys.argv[1]
	img_name, ext = file_name.split(".")
	rel_path = "result/" +img_name + "/"
	img_source_path = "../../img/" + file_name

	if os.path.isfile(img_source_path) != 1:
		print("File (",img_source_path,") does not exist!")
		return

	img=Image.open(img_source_path)
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

	if(os.path.isdir("result/" + img_name) != 1):
		call(["mkdir","result/" + img_name])

	resulting_energy(img,seg)

	output_file_name = str(img_name) + "_out." + str(ext)
	file_path = rel_path + output_file_name
	scipy.misc.imsave(file_path,seg*255)


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




if __name__=="__main__":
	main()	
