import cv2
import numpy as np
import argparse
import imutils
import time
import os
from tracker import Sort
from utils import *
from PIL import Image
from time import sleep
from imutils.video import FPS
import pytesseract
import csv
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


counter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-d", "--digits", type=int, default=1,
        help="whether or not *digits only* OCR will be performed")
args = vars(ap.parse_args())

with open('13012022.csv', 'w', newline='') as f:
                fw = csv.writer(f)
                fw.writerow(['IDvehicle','Vehicle type', 'Date', 'Time'])
  
    
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["./yolov3", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["./yolov3", "yolov3.weights"])
configPath = os.path.sep.join(["./yolov3", "yolov3.cfg"])

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# and determine only the *output* layer names that we need from YOLO
print("loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")

frameIndex = 0
memory = {}
tracker = Sort()
detec = []
offset = 5
roi = 210
# loop over frames from the video
while True:
	# read the next frame from the video
	(read, frame) = vs.read()


	
	if not read: #in case the frame was not read means that the video has ended
		break 

	# should be true only for the first frame
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	# blob is the preprocessed(scaled, resized) image frame
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob) #giving the model the frame as input
	
	layerOutputs = net.forward(ln) #model return the outputs 
	

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if (LABELS[classID] == ("car")or LABELS[classID] == ("truck")or LABELS[classID] == ("bus")) and confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
	
	dets = []
	
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			

			# draw a bounding box rectangle and label on the image
					
			#color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			#cv2.rectangle(frame, (x, y), (w, h), color, 2)

			centro = pega_centro(x, y, w, h)
			detec.append(centro)
			
			for (x,y) in detec:
                            
                            if y<(roi+offset) and y>(roi-offset):
                                counter += 1
                                vehicle = LABELS[classIDs[i]]
                                detec.remove((x,y))
                                cv2.imwrite("frame.jpg", frame)
                
                                # OCR date and time
                                imaged = Image.open("frame.jpg")
                                boxd = (0, 0, 235, 50)
                                cropped_imaged = imaged.crop(boxd)
                                cropped_imaged.save('digitd.png')           
               
   
                                imagedd = cv2.imread('digitd.png')
                                img = cv2.cvtColor(imagedd, cv2.COLOR_BGR2GRAY)
                                ret, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
                                cv2.imwrite('date.jpg', thresh2)
          
          
                                image = Image.open("frame.jpg")
                                box = (249, 0, 421, 50)
                                cropped_image = image.crop(box)
                                digit1 = (0, 0, 51, 50)
                                cropped_image2 = cropped_image.crop(digit1)
                                cropped_image2.save('digit1.png')
                                digit2 = (60, 0, 111, 78)
                                cropped_image2 = cropped_image.crop(digit2)
                                cropped_image2.save('digit2.png')
                                digit3 = (120, 0, 171, 78)
                                cropped_image2 = cropped_image.crop(digit3)
                                cropped_image2.save('digit3.png')

                                image = Image.open('digitt.png')
                                num1 = Image.open('digit1.png')
                                image_copy = image.copy()
                                position = (0, 0)
                                image_copy.paste(num1, position)
                                image_copy.save('digittt.png')

                                image2 = Image.open('digittt.png')
                                num2 = Image.open('digit2.png')
                                image2_copy = image2.copy()
                                position = (90, 0)
                                image_copy.paste(num2, position)
                                image_copy.save('digittt.png')

                                image3 = Image.open('digittt.png')
                                num3 = Image.open('digit3.png')
                                image3_copy = image3.copy()
                                position = (178, 0)
                                image_copy.paste(num3, position)
                                image_copy.save('digittt.png')
      
                                image4 = cv2.imread('digittt.png')
                                img = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
                                ret, thresh3 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
                                cv2.imwrite('test5.jpg', thresh3)

                
                                image = cv2.imread('date.jpg')
                                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
                                # check to see if *digit only* OCR should be performed, and if so,
                                # update our Tesseract OCR options
                                if args["digits"] > 0:
                                      options = "outputbase digits"
                                # OCR the input image using Tesseract
                                textd = pytesseract.image_to_string(rgb, config=options)
          
          
                                image = cv2.imread('digittt.png')
                                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
                                # check to see if *digit only* OCR should be performed, and if so,
                                # update our Tesseract OCR options
                                if args["digits"] > 0:
                                      options = "outputbase digits"
                                # OCR the input image using Tesseract
                                textt = pytesseract.image_to_string(rgb, config=options)
                                print("Vehicle type",vehicle,"date",textd[0:10],"time",textt[0:6])

                                with open('13012022.csv', 'a', newline='') as f:
                                        fw = csv.writer(f)
                                        fw.writerow([counter, LABELS[classIDs[i]], textd[0:10], textt[0:6]])
vs.release()
