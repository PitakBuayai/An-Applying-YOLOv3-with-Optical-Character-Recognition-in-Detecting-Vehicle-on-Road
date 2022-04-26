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



line = [(494, 494), (1550, 700)] #use test2
#line = [(390, 200), (1600, 390)]
counter = 0
viheclecounter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-d", "--digits", type=int, default=1,
        help="whether or not *digits only* OCR will be performed")
                

args = vars(ap.parse_args())

with open('vi3.csv', 'w', newline='') as f:
                fw = csv.writer(f)
                fw.writerow(['IDvehicle','Vehicle type', 'Date', 'Time'])



# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["./yolov3", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["./yolov3", "yolov3.weights"])
configPath = os.path.sep.join(["./yolov3", "yolov3.cfg"])

# initialize a list of colors to represent each possible class label
np.random.seed(0)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# and determine only the *output* layer names that we need from YOLO
print("loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
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
compare = 0

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
	start = time.time()
	layerOutputs = net.forward(ln) #model return the outputs 
	end = time.time()

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
			if confidence > args["confidence"]:
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
	if len(idxs) > 0 :
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	tracks = tracker.update(dets)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}

	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			#cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			#color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			cv2.rectangle(frame, (x, y), (w, h), color, 2)

			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				cv2.line(frame, p0, p1, color, 3)

				if intersect(p0, p1, line[0], line[1]):
                                        vehicle = LABELS[classIDs[i]]
                                        counter += 1

			#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text = "{}".format(indexIDs[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			i += 1

	# line
	cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
	if counter > 0 and counter != compare :
                viheclecounter += 1

                cv2.imwrite("frame.jpg", frame)
                # OCR date and time
                imaged = Image.open("frame.jpg")
                boxd = (0, 4, 218, 34)
                cropped_imaged = imaged.crop(boxd)

                digitd = (0, 2, 42, 29)
                cropped_imaged2 = cropped_imaged.crop(digitd)
                cropped_imaged2.save('digitd.png')

                digitm = (67, 2, 108, 29)
                cropped_imaged2 = cropped_imaged.crop(digitm)
                cropped_imaged2.save('digitm.png')

                digity = (132, 2, 218, 29)
                cropped_imaged2 = cropped_imaged.crop(digity)
                cropped_imaged2.save('digity.png')


                imaged = Image.open('550.jpg')
                numd1 = Image.open('digitd.png')
                image_copy = imaged.copy()
                position = (238, 270)
                image_copy.paste(numd1, position)
                image_copy.save('date.jpg')

                imaged2 = Image.open('date.jpg')
                numd2 = Image.open('digitm.png')
                image2_copy = imaged2.copy()
                position = (282, 270)
                image_copy.paste(numd2, position)
                image_copy.save('date.jpg')

                imaged3 = Image.open('date.jpg')
                numd3 = Image.open('digity.png')
                image3_copy = imaged3.copy()
                position = (329, 270)
                image_copy.paste(numd3, position)
                image_copy.save('date.jpg')


                image4 = cv2.imread('date.jpg')
                img = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
                ret, thresh2 = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
                ret, thresh3 = cv2.threshold(thresh2, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
                cv2.imwrite('date.jpg', thresh3)
          
          
                image = Image.open("frame.jpg")
                box = (241, 4, 417, 34)
                cropped_image = image.crop(box)
                digit1 = (2, 2, 42, 29)
                cropped_image2 = cropped_image.crop(digit1)
                cropped_image2.save('digit1.png')

                digit2 = (65, 2, 110, 29)
                cropped_image2 = cropped_image.crop(digit2)
                cropped_image2.save('digit2.png')

                digit3 = (133, 2, 175, 29)
                cropped_image2 = cropped_image.crop(digit3)
                cropped_image2.save('digit3.png')
             

                image = Image.open('550.jpg')
                num1 = Image.open('digit1.png')
                image_copy = image.copy()
                position = (238, 270)
                image_copy.paste(num1, position)
                image_copy.save('test5.jpg')

                image2 = Image.open('test5.jpg')
                num2 = Image.open('digit2.png')
                image2_copy = image2.copy()
                position = (280, 270)
                image_copy.paste(num2, position)
                image_copy.save('test5.jpg')

                image3 = Image.open('test5.jpg')
                num3 = Image.open('digit3.png')
                image3_copy = image3.copy()
                position = (326, 270)
                image_copy.paste(num3, position)
                image_copy.save('test5.jpg')


                image4 = cv2.imread('test5.jpg')
                img = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
                ret, thresh2 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
                ret, thresh3 = cv2.threshold(thresh2, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
                cv2.imwrite('test5.jpg', thresh3)

                
                image = cv2.imread('date.jpg')
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
                # check to see if *digit only* OCR should be performed, and if so,
                # update our Tesseract OCR options
                if args["digits"] > 0:
                          options = "outputbase digits"
                # OCR the input image using Tesseract
                textd = pytesseract.image_to_string(rgb, config=options)
          
          
                image = cv2.imread('test5.jpg')
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
                # check to see if *digit only* OCR should be performed, and if so,
                # update our Tesseract OCR options
                if args["digits"] > 0:
                          options = "outputbase digits"
                # OCR the input image using Tesseract
                textt = pytesseract.image_to_string(rgb, config=options)
                print("Vehicle",viheclecounter,"date",textd[0:8],"time",textt[0:6])        
                compare = counter

                with open('vi3.csv', 'a', newline='') as f:
                        fw = csv.writer(f)
                        fw.writerow([viheclecounter, vehicle, textd[0:8], textt[0:6]])
                compare = counter

	# counter
	cv2.putText(frame, str(viheclecounter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)


	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1

	if frameIndex >= 4000:
		print("[INFO] cleaning up...")
		writer.release()
		vs.release()
		exit()

# release the file pointers 
print("cleaning up...")
writer.release()
vs.release()
