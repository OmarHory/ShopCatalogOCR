import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import argparse


def get_bounding_box(f):
	boxes_list = []
	with open(f,'r') as txt:
		for i in txt:
			boxes_list.append(i.split(" "))
		return boxes_list

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,help="path to input image")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=640,help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=960,help="resized image height (should be multiple of 32)")
ap.add_argument("-t", "--txt", type=str,help="bounding box file")

args = vars(ap.parse_args())

image = cv2.imread(args['image'])
# plt.imshow(image)
# plt.show()

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Saving a original image and shape
orig = image.copy()
print(image.shape[:2])
# (origH, origW) = image.shape[:2]

# # set the new height and width to default 320 by using args #dictionary.  
# (newW, newH) = (args["width"], args["height"])

# #Calculate the ratio between original and new image for both height and weight. 
# #This ratio will be used to translate bounding box location on the original image. 
# rW = origW / float(newW)
# rH = origH / float(newH)

# # resize the original image to new dimensions
# image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# # construct a blob from the image to forward pass it to EAST model
# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
# 	(123.68, 116.78, 103.94), swapRB=True, crop=False)

# layerNames = [
#     "feature_fusion/Conv_7/Sigmoid",
#     "feature_fusion/concat_3"]

# net = cv2.dnn.readNet(args["east"])
# #Forward pass the blob from the image to get the desired output layers
# net.setInput(blob)
# (scores, geometry) = net.forward(layerNames)


# ## Returns a bounding box and probability score if it is more than minimum confidence
# def predictions(prob_score, geo):
# 	(numR, numC) = prob_score.shape[2:4]
# 	boxes = []
# 	confidence_val = []

# 	# loop over rows
# 	for y in range(0, numR):
# 		scoresData = prob_score[0, 0, y]
# 		x0 = geo[0, 0, y]
# 		x1 = geo[0, 1, y]
# 		x2 = geo[0, 2, y]
# 		x3 = geo[0, 3, y]
# 		anglesData = geo[0, 4, y]

# 		# loop over the number of columns
# 		for i in range(0, numC):
# 			if scoresData[i] < args["min_confidence"]:
# 				continue

# 			(offX, offY) = (i * 4.0, y * 4.0)

# 			# extracting the rotation angle for the prediction and computing the sine and cosine
# 			angle = anglesData[i]
# 			cos = np.cos(angle)
# 			sin = np.sin(angle)

# 			# using the geo volume to get the dimensions of the bounding box
# 			h = x0[i] + x2[i]
# 			w = x1[i] + x3[i]

# 			# compute start and end for the text pred bbox
# 			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
# 			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
# 			startX = int(endX - w)
# 			startY = int(endY - h)

# 			boxes.append((startX, startY, endX, endY))
# 			confidence_val.append(scoresData[i])

# 	# return bounding boxes and associated confidence_val
# 	return (boxes, confidence_val)


# # Find predictions and  apply non-maxima suppression
# (boxes, confidence_val) = predictions(scores, geometry)
# boxes = non_max_suppression(np.array(boxes), probs=confidence_val)


##Text Detection and Recognition 

# initialize the list of results
results = []


boxes = get_bounding_box(args['txt'])
for (_, centerX, centerY, boxW, boxH) in boxes:

	centerX = float(centerX.rstrip() ) * W
	centerY = float(centerY.rstrip() ) * H
	boxW = float(boxW.rstrip() ) * W
	boxH = float(boxH.rstrip() ) * H

	startX = int(centerX - (boxW/2))
	startY = int(centerY - (boxH/2))
	endX = int(centerX + (boxW/2))
	endY = int(centerY + (boxH/2))
	

	
	# import pdb
	# pdb.set_trace()
	
	try:
		#extract the region of interest
		r = orig[startY:endY, startX:endX]

		#configuration setting to convert image to string.  
		configuration = ("-l eng --oem 1 --psm 11")
		##This will recognize the text from the image of bounding box
		
		text = pytesseract.image_to_string(r, config=configuration)

		# append bbox coordinate and associated text to the list of results 
		results.append(((startX, startY, endX, endY), text))
	except:
		pass



#Display the image with bounding box and recognized text
orig_image = orig.copy()

# print(results)
# import pdb
# pdb.set_trace()

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
	print("{}\n".format(text))

	# Displaying text
	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 0, 255), 2)
	cv2.putText(orig_image, "", (start_X, start_Y - 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

plt.imshow(orig_image)
plt.title('Output')
plt.show()


