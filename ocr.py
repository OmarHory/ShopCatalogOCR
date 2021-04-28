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

orig = image.copy()
print(image.shape[:2])
(H, W) = image.shape[:2]

results = []

#TODO: make an api class to support YOLOv5 with tessaract
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



orig_image = orig.copy()

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
