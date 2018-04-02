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
import sys
from dwave_qbsolv import QBSolv


#total_cost = 1000000
#threshold = 2000
def main():
	# Read in image
	file_name = sys.argv[1]
	global total_cost
	old_total_cost = 0
	img=Image.open(file_name)
	img=numpy.array(img)
	
	qubo_extractor(img,file_name)
	

def qubo_extractor(img,file_name):
	img_name, ext = file_name.split(".")
	seg_file_name = img_name + "_segmentation.qubo"
	#f = open(seg_file_name, 'w')
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
	output_file_name = str(img_name) + "_out_python." + str(ext)
	scipy.misc.imsave(output_file_name,out*255)

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



if __name__=="__main__":
	main()	
