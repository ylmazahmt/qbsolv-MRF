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
from dwave_qbsolv import QBSolv

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
args = cmd_folder.split('/')
src_folder = '/'.join(args[:-1])
sys.path.insert(0, (src_folder))

from MRF.mrf import *
from MRF.superpixel import *

def main():
	# Read in image
	
	file_name = sys.argv[1]
	img_name, ext = file_name.split(".")
	rel_path = "result/" +img_name + "/"
	img_source_path = "../../img/" + file_name
	img_model_path = "../../img/" + img_name + "_model"
	
	if os.path.isfile(img_source_path) != 1:
		print("File (",img_source_path,") does not exist!")
		return

	image=Image.open(img_source_path)
	img=numpy.array(image)

	# Create Superpixels and Model 
	superpixels,segNeighbors,segments,uniqueCouplers,segDict = superpixel_extractor(img)
	foregroundModel, backgroundModel = model_extractor(image,img_model_path)

	qubo_extractor(superpixels, segNeighbors, uniqueCouplers, foregroundModel, backgroundModel, file_name, rel_path, img)
	#scipy.misc.imsave('out.png',seg*255)

def qubo_extractor(superpixels, segNeighbors, uniqueCouplers, foregroundModel, backgroundModel, file_name, rel_path, img):
	Q = {}
	N = len(superpixels)
	for i in range(N):
		foreground_node_id = (i)*2
		background_node_id = foreground_node_id+1
		Q[(foreground_node_id,foreground_node_id)] = unary_potential(superpixels,1,i,foregroundModel,backgroundModel)
		#background qubit
		Q[(background_node_id,background_node_id)] = unary_potential(superpixels,0,i,foregroundModel,backgroundModel)
		#for qubits of same pixel - high cost should be given here to ensure both qubits are not open
		Q[(foreground_node_id,background_node_id)] = 10
		
		#for neighbors
	for coupler in uniqueCouplers:
		leftCoupler = coupler[0]
		rightCoupler = coupler[1]
		foreground_node_id = leftCoupler*2
		background_node_id = foreground_node_id+1
		neighbor_foreground_node_id = rightCoupler*2
		neighbor_background_node_id = neighbor_foreground_node_id+1
		
		Q[(foreground_node_id, neighbor_foreground_node_id)] = doubleton_potential(superpixels,1,1,leftCoupler,rightCoupler)
		Q[(background_node_id, neighbor_foreground_node_id)] = doubleton_potential(superpixels,0,1,leftCoupler,rightCoupler)
		Q[(foreground_node_id, neighbor_background_node_id)] = doubleton_potential(superpixels,1,0,leftCoupler,rightCoupler)
		Q[(background_node_id, neighbor_background_node_id)] = doubleton_potential(superpixels,0,0,leftCoupler,rightCoupler)
		
	

	response = QBSolv().sample_qubo(Q)
	result = list(response.samples())
	array = result[0]

	(M,N) = img.shape[0:2]
	rgb = img.shape[2:3]

	# is image grayscale
	isGrayscale = 1
	if len(rgb) > 0:
		isGrayscale = 0

	segments = getSegments(img,isGrayscale)
	counter = 0
	seg = []
	for i in array:
		mode = counter
		if(mode%2 == 0):
			if array[i] == 1:
				seg.append(1)
			else:
				seg.append(0)
		counter += 1

	seg= numpy.array(seg)
	if isGrayscale:
		output_image = numpy.zeros(shape=(M,N))
	else:
		output_image = numpy.zeros(shape=(M,N,3))
	for i in range(M):
		for j in range(N):
			if seg[segments[i,j]] == 1:
				if isGrayscale:
					output_image[i,j] = img[i,j]
				else:
					output_image[i,j,0:3] = img[i,j,0:3]
	
	img_name, ext = file_name.split(".")

	if(os.path.isdir("result/" + img_name) != 1):
		call(["mkdir","result/" + img_name])

	output_file_name = str(img_name) + "_out_python." + str(ext)
	file_path = rel_path + output_file_name
	scipy.misc.imsave(file_path,output_image*255)

	if(os.path.isfile(file_path)):
		call(["open", file_path])
		return

if __name__=="__main__":
	main()	
