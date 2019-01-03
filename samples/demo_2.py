import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
							 'bus', 'train', 'truck', 'boat', 'traffic light',
							 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
							 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
							 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
							 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
							 'kite', 'baseball bat', 'baseball glove', 'skateboard',
							 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
							 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
							 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
							 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
							 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
							 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
							 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
							 'teddy bear', 'hair drier', 'toothbrush']
							 
							 
def process_images(images, names, out_dir):
	results = model.detect(images, verbose=0)

	# Visualize results
	for j in range(config.BATCH_SIZE):
		r = results[j]
		visualize.display_instances(images[j], names[j], r['rois'], r['masks'], r['class_ids'], 
										class_names, r['scores'], out_dir=out_dir)						 
							 

image_dir = '/home/kandhavarapu/Projects/bianchini/data/liquor_data'
out_dir = '/home/kandhavarapu/Projects/bianchini/data/segmented_out'
processed = "/home/kandhavarapu/Projects/bianchini/data/processed.txt"

if __name__ == "__main__":
	with open(processed) as f:
		processed_files = f.readlines()[0].strip().split(",")

	import cv2
	i = 0
	images = []
	names = []
	for file in sorted(os.listdir(image_dir))[5000:]:
		
		if file[:-4]+".png" in processed_files:
			continue
		if os.path.exists(os.path.join(out_dir, file[:-4]+".png")):
			continue
		filename = os.path.join(image_dir,file)
		image = cv2.imread(filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Run detection
		if i < config.BATCH_SIZE:
			#print("reading batch of 100!!")
			images.append(image)
			names.append(file)
			i += 1
		else:
			process_images(images, names, out_dir)
			i = 0
			images = []
			names = []
	if len(images) > 0:
		process_images(images,names)

					 
					 
							 
							 
							 
							 
							 
