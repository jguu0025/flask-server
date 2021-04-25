'''edit bu Jiayu Gu
15/4/2021'''

from flask import Flask
from flask import request
import base64
import numpy as np
import sys
import time
import cv2
import os
import json

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello!"

def get_labels(labels_path):

    lpath = os.path.sep.join([labels_path])
    # print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([config_path])
    return configPath


def load_model(configPath, weightsPath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    # Exception name 'configpath' is not defined 
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net


def do_prediction(image, net, LABELS):
    # construct the argument parse and parse the arguments
    confthres = 0.3
    nmsthres = 0.1


    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]
    
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)
    

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        object_list = []

        for i in idxs.flatten():
            objects_dic = {"label":LABELS[classIDs[i]], "accuracy": confidences[i], "rectangle":{ "height": boxes[i][0], "left": boxes[i][1], "top": boxes[i][2], "width": boxes[i][3]}
            }
            object_list.append(objects_dic)
        return object_list

@app.route('/api/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        
        # Yolov3-tiny versrion
        labelsPath = "yolo_tiny_configs/coco.names"
        cfgpath = "yolo_tiny_configs/yolov3-tiny.cfg"
        wpath = "yolo_tiny_configs/yolov3-tiny.weights"

        Lables = get_labels(labelsPath)
        CFG = get_config(cfgpath)
        Weights = get_weights(wpath)
        
        respond_dic = {}
        request_data = request.json
        request_data_json = json.loads(request_data)
        # decode base 64 image
        id = request_data_json['id']
        request_image = request_data_json['image']
        request_image_decoded = base64.b64decode(request_image)
        path = "images/"
        parent_path = os.getcwd()
        # create a file to hold images
        save_parent_path = os.path.join(parent_path, path)
        if not os.path.exists(save_parent_path):
            os.mkdir(save_parent_path)
        filename =save_parent_path + str(id) + '.jpg'
        
        with open(filename, 'wb') as file1:
            file1.write(request_image_decoded)

        try:
            imagefile = str(filename) #filename has to be the whole directory including filename
            img = cv2.imread(imagefile)
            npimg = np.array(img)
            image = npimg.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # load the neural net.  Should be local to this method as its multi-threaded endpoint
            nets = load_model(CFG, Weights)
            respond_dic['id'] = id
            respond_dic['objects'] = do_prediction(image, nets, Lables)
            json_str = json.dumps(respond_dic, indent=4)
            return json_str

        except Exception as e:
            return_str = "Exception {}".format(e)
            print(return_str)
            return return_str

    if request.method == 'GET':
        return "ERROR method! Please post images."


if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=True)
