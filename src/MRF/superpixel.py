# import the necessary packages
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import scipy

numSegments = 10

# def model_extractor(image,img_model_path):
# 	img_model_path = img_model_path + ".txt"
# 	counter = 0
# 	img_model = []
# 	with open(img_model_path, "r") as ins:
# 		for line in ins:
# 			img_model.append(int(line))
# 	print("Image Model Indices: ",img_model)
# 	foregroundModel = image.crop((img_model[0], img_model[1], img_model[2], img_model[3]))
# 	scipy.misc.imsave("fg.png",foregroundModel)
# 	backgroundModel = image.crop((img_model[4], img_model[5], img_model[6], img_model[7]))
# 	scipy.misc.imsave("bg.png",backgroundModel)

# 	foregroundModel = np.array(foregroundModel)
# 	backgroundModel = np.array(backgroundModel)
# 	print("Foreground Model:", foregroundModel)
# 	print("Background Model:", backgroundModel)
# 	return foregroundModel, backgroundModel

def model_extractor(image, img_model_path):
	fg_path = img_model_path + "_fg.png"
	bg_path = img_model_path + "_bg.png"
	fg_img = Image.open(fg_path)
	bg_img = Image.open(bg_path)

	foregroundModel = np.array(fg_img)
	backgroundModel = np.array(bg_img)

	return foregroundModel, backgroundModel

def getSegments(image, isGrayscale):
	if isGrayscale:
		image = np.dstack([image, image, image])
	segments = slic(image, n_segments = numSegments, sigma = 5)
	return segments

def superpixel_extractor(image):
	(M,N)=image.shape[0:2]
	segDict = dict()
	segNeighbors = dict()
	superpixels = dict()
	
	img = image
	
	# if photo is grayscale
	rgb = image.shape[2:3]
	isGrayscale = 1
	if len(rgb) > 0:
		isGrayscale = 0

	segments = getSegments(image,isGrayscale)
	print(segments)
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
	 
	# show the plots
	plt.show()

	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):

		# # construct a mask for the segment
		# print("i:",i)
		# print("segVal:", segVal)
		# print ("[x] inspecting segment %d" % (i))
		# mask = np.zeros(image.shape[:2], dtype = "uint8")
		# # print(mask)
		# mask[segments == segVal] = 255
		
		segDict[segVal] = [(j,k) for j in range(M) for k in range(N) if segments[j,k] == segVal]
		
		superpixels[segVal] = [img[j,k] for j in range(M) for k in range(N) if segments[j,k] == segVal]
		superpixels[segVal] = np.array(superpixels[segVal])
		
		segNeighbors[segVal] = []
		for pair in segDict[segVal]:
			if segments[pair[0]-1,pair[1]-1] != segVal and pair[0]-1 >= 0 and pair[1]-1 >= 0:
				segNeighbors[segVal].append(segments[pair[0]-1,pair[1]-1])
			elif segments[pair[0]-1,pair[1]] != segVal and pair[0]-1 >= 0 and pair[1] >= 0:
				segNeighbors[segVal].append(segments[pair[0]-1,pair[1]])
			elif pair[1]+1 < N and segments[pair[0]-1,pair[1]+1] != segVal and pair[0]-1 >= 0:
				segNeighbors[segVal].append(segments[pair[0]-1,pair[1]+1])

			elif segments[pair[0],pair[1]-1] != segVal and pair[0] >= 0 and pair[1]-1 >= 0:
				segNeighbors[segVal].append(segments[pair[0],pair[1]-1])
			elif pair[1]+1 < N and segments[pair[0],pair[1]+1] != segVal and pair[0] >= 0:
				segNeighbors[segVal].append(segments[pair[0],pair[1]+1])

			elif pair[0]+1 < M and segments[pair[0]+1,pair[1]-1] != segVal and pair[1]-1 >= 0:
				segNeighbors[segVal].append(segments[pair[0]+1,pair[1]-1])
			elif pair[0]+1 < M and segments[pair[0]+1,pair[1]] != segVal and pair[1] >= 0:
				segNeighbors[segVal].append(segments[pair[0]+1,pair[1]])
			elif  pair[0]+1 < M and pair[1]+1 < N and segments[pair[0]+1,pair[1]+1] != segVal:
				segNeighbors[segVal].append(segments[pair[0]+1,pair[1]+1])
		segNeighbors[segVal] = np.unique(np.asarray(segNeighbors[segVal]))
		
		
		# # show the masked region
		# cv2.imshow("Mask", mask)
		# cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
		# cv2.waitKey(0)
	
	uniqueCouplers = []
	for i,_ in enumerate(segNeighbors):
		uniqueCouplers.append([(i,neighbor) for neighbor in segNeighbors[i] if neighbor > i])
	uniqueCouplers = [item for sublist in uniqueCouplers for item in sublist]

	# print(uniqueCouplers)

	return superpixels,segNeighbors,segments,uniqueCouplers,segDict
