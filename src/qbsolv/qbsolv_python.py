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

def main():
	# Read in image
	img_source_path = "../../img/" + sys.argv[1]
	args = img_source_path.split("/")
	file_name = args[-1]
	global total_cost
	old_total_cost = 0
	img=Image.open(img_source_path)
	img=numpy.array(img)
	
	qubo_extractor(img,file_name)
	

def qubo_extractor(img,file_name):
	img_name, ext = file_name.split(".")
	(M,N) = img.shape[0:2]
	coupler_count = 0
	Q = {}
	for i in range(M):
		for j in range(N):
			foreground_node_id = (i*N*2)+(j*2)
			background_node_id = foreground_node_id+1
			#foreground qubit
			Q[(foreground_node_id,foreground_node_id)] = unary_potential(img,1,i,j)
			#background qubit
			Q[(background_node_id,background_node_id)] = unary_potential(img,0,i,j)
			#for qubits of same pixel - high cost should be given here to ensure both qubits are not open
			Q[(foreground_node_id,background_node_id)] = 10
			coupler_count += 1
			#for neighbors
			neighbors = find_neighbors(img,i,j)
			for k in neighbors:
				neighbor_foreground_node_id = (k[0]*N*2)+(k[1]*2)
				neighbor_background_node_id = neighbor_foreground_node_id+1
				#f.write('  '+ str(foreground_node_id) +' '+ str(neighbor_foreground_node_id) +' '+ str(f_1(img,i,j,k)) +'\n' )
				#f.write('  '+ str(background_node_id) +' '+ str(neighbor_foreground_node_id) +' '+ str(f_2(img,i,j,k)) +'\n' )
				#f.write('  '+ str(foreground_node_id) +' '+ str(neighbor_background_node_id) +' '+ str(f_2(img,i,j,k)) +'\n' )
				#f.write('  '+ str(background_node_id) +' '+ str(neighbor_background_node_id) +' '+ str(f_1(img,i,j,k)) +'\n' )
				#coupler_count += 1

	#print(Q)			
	
	
	response = QBSolv().sample_qubo(Q)
	result = list(response.samples())

	counter = 1
	out = []
	array = result[0]
	for i in array:
		mode = counter
		if(counter > (N*M*2)):
			continue
		if(mode%2 == 1):
			if array[i] == 1:
				out.append(int(255))
			else:
				out.append(0)
		counter += 1

		
	out=numpy.array(out)
	out = out.reshape(M,N)
	img_name, ext = file_name.split(".")

	if(os.path.isdir("result/" + img_name) != 1):
		call(["mkdir","result/" + img_name])

	rel_path = "result/" +img_name + "/"
	output_file_name = str(img_name) + "_out_python." + str(ext)
	file_path = rel_path + output_file_name
	scipy.misc.imsave(file_path,out)


if __name__=="__main__":
	main()	
