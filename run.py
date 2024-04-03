import cv2 as cv
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

model_path=''
labelmap_path=''

model = tf.saved_model.load(model_path)
categoryIndex=label_map_util.create_category_index_from_labelmap(labelmap_path,use_display_name=True)

getFrame=np.zeros((2,2)) ####################### <-- FRAME TO RUN INFERENCE ON ########################
frame=getFrame.copy()

frame=np.asarray(frame)
inputTensor=tf.convert_to_tensor(frame)
inputTensor=inputTensor[tf.newaxis,...]
outputDict=model(inputTensor)

numDetections=int(outputDict.pop('num_detections'))
outputDict={key:value[0,:numDetections].numpy()
            for key,value in outputDict.items()}
outputDict['num_detections']=numDetections
outputDict['detection_classes']=outputDict['detection_classes'].astype(np.int64)

if 'detection_masks' in outputDict:
    reframeDM=utils_ops.reframe_box_masks_to_image_masks(
        outputDict['detection_masks'],outputDict['detection_boxes'],
        frame.shape[0],frame.shape[1])
    reframeDM=tf.cast(reframeDM>.5,tf.uint8)
    outputDict['detection_masks_reframed']=reframeDM.numpy()

if outputDict:
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        outputDict['detection_boxes'],
        outputDict['detection_classes'],
        outputDict['detection_scores'],
        categoryIndex,
        instance_masks=outputDict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=8
    )
    cv.imshow('Detection',frame)
