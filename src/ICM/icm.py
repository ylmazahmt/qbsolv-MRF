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
from MRF.superpixel import *

total_cost = 1000000
threshold = 0.4
def main():
	# Read in image
	global total_cost
	old_total_cost = 0

	file_name = sys.argv[1]
	img_name, ext = file_name.split(".")
	rel_path = "result/" +img_name + "/"
	img_source_path = "../../img/" + file_name
	img_model_path = "../../img/" + img_name + "_model"
	if os.path.isfile(img_source_path) != 1:
		print("File (",img_source_path,") does not exist!")
		return


	image=Image.open(img_source_path)
	# image_1 = image.convert('LA')
	# image_1.save('greyscale.png')
	img=numpy.array(image)
	
	# Create Superpixels and Model 
	superpixels,segNeighbors,segments,uniqueCouplers,segDict = superpixel_extractor(img)
	foregroundModel, backgroundModel = model_extractor(image,img_model_path)
	
	(M,N)=img.shape[0:2]
	rgb = img.shape[2:3]
	print(M,N,rgb)
	
	# is image grayscale
	isGrayscale = 1
	if len(rgb) > 0:
		isGrayscale = 0
	
	seg = init_config(superpixels)
	while(abs(total_cost - old_total_cost) > threshold):
		old_total_cost = total_cost
		seg = ICM(superpixels,seg,segNeighbors,foregroundModel,backgroundModel)
		print(total_cost)

	if(os.path.isdir("result/" + img_name) != 1):
		call(["mkdir","result/" + img_name])

	if isGrayscale:
		output_image = numpy.zeros(shape=(M,N))
	else:
		output_image = numpy.zeros(shape=(M,N,3))
	output_file_name = str(img_name) + "_out." + str(ext)
	file_path = rel_path + output_file_name
	for i in range(M):
		for j in range(N):
			if seg[segments[i,j]] == 1:
				if isGrayscale:
					output_image[i,j] = img[i,j]
				else:
					output_image[i,j,0:3] = img[i,j,0:3]
	
	resulting_energy(superpixels,seg,segNeighbors,foregroundModel,backgroundModel)

	scipy.misc.imsave(file_path,output_image*255)

	if(os.path.isfile(file_path)):
		call(["open", file_path])
		return

def init_config(superpixels):
	(N)= len(superpixels)
	print("Segment Count:",N)
	img_seg = numpy.zeros(shape=(N))
	for i in range(N):
		img_seg[i] = randint(0, 1)
	print("img_seg",img_seg)
	return img_seg

def ICM(superpixels,seg,segNeighbors,foregroundModel,backgroundModel):
	global total_cost
	total_cost = 0
	(N)= len(superpixels)
	for i in range(N):
		# Find segmentation level which has min energy (highest posterior)
		cost=[energy(superpixels,seg,k,i,segNeighbors,foregroundModel,backgroundModel) for k in range(2)]
		total_cost += min(cost)
		#print (total_cost)
		seg[i]=cost.index(min(cost))
	return seg




if __name__=="__main__":
	main()	
