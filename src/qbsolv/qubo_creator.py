# Image segmentation using MRF model with qbsolv
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
from MRF.superpixel import *

def main():
	# Read in image
	img_source_path = sys.argv[1]
	args = img_source_path.split("/")
	file_name = args[-1]
	img_name, ext = file_name.split(".")
	img_model_path = "../../img/" + img_name + "_model.txt"
	
	image=Image.open(img_source_path)
	img=numpy.array(image)

	# Create Superpixels and Model 
	superpixels,segNeighbors,segments,uniqueCouplers,segDict = superpixel_extractor(img)
	foregroundModel, backgroundModel = model_extractor(image,img_model_path)

	qubo_extractor(superpixels, segNeighbors, uniqueCouplers, foregroundModel, backgroundModel, img_name)
	#scipy.misc.imsave('out.png',seg*255)

def qubo_extractor(superpixels, segNeighbors, uniqueCouplers, foregroundModel, backgroundModel, img_name):
	seg_file_name = img_name + "_segmentation.qubo"
	rel_path = "result/" +img_name + "/" + seg_file_name
	f = open(rel_path,'w')
	N = len(superpixels)
	coupler_count = 0

	for i in range(N):
		f.write('c for superpixel - '+str(i)+':\n')
		foreground_node_id = (i)*2
		background_node_id = foreground_node_id+1
		#foreground qubit
		f.write('  '+ str(foreground_node_id) +' '+ str(foreground_node_id) +' '+ str(unary_potential(superpixels,1,i,foregroundModel,backgroundModel)) +'\n' )
		#background qubit
		f.write('  '+ str(background_node_id) +' '+ str(background_node_id) +' '+ str(unary_potential(superpixels,0,i,foregroundModel,backgroundModel)) +'\n' )
		#for qubits of same pixel - high cost should be given here to ensure both qubits are not open
		f.write('  '+ str(foreground_node_id) +' '+ str(background_node_id) +' '+ str(10) +'\n' )
		coupler_count += 1
		#for neighbors
		
	for coupler in uniqueCouplers:
		leftCoupler = coupler[0]
		rightCoupler = coupler[1]
		foreground_node_id = leftCoupler*2
		background_node_id = foreground_node_id+1
		neighbor_foreground_node_id = rightCoupler*2
		neighbor_background_node_id = neighbor_foreground_node_id+1
		# print("Left Coupler:",leftCoupler,"Right Coupler:", rightCoupler)
		f.write('c coupler between superpixels '+str(leftCoupler)+ " and "+str(rightCoupler)+':\n')
		f.write('  '+ str(foreground_node_id) +' '+ str(neighbor_foreground_node_id) +' '+ str(doubleton_potential(superpixels,1,1,leftCoupler,rightCoupler)) +'\n' )
		f.write('  '+ str(background_node_id) +' '+ str(neighbor_foreground_node_id) +' '+ str(doubleton_potential(superpixels,0,1,leftCoupler,rightCoupler)) +'\n' )
		f.write('  '+ str(foreground_node_id) +' '+ str(neighbor_background_node_id) +' '+ str(doubleton_potential(superpixels,1,0,leftCoupler,rightCoupler)) +'\n' )
		f.write('  '+ str(background_node_id) +' '+ str(neighbor_background_node_id) +' '+ str(doubleton_potential(superpixels,0,0,leftCoupler,rightCoupler)) +'\n' )
		coupler_count += 4

	#to add first line information			
	f.close()
	f = open(rel_path,'r+')
	content = f.read() # read old content
	f.seek(0) # go back to the beginning of the file
	f.write('p  '+ 'qubo  ' + '0  ' + str(N*2) + ' ' + str(N*2) + ' ' + str(coupler_count) + '\n')
	f.write(content)
	f.close()



if __name__=="__main__":
	main()	
