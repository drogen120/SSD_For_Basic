import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
# import xml.etree.ElementTree as ET
from lxml import etree as ET
slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('./')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

IMAGE_FOLDER = "./owndata/traffic_img/test/"
LABEL_FOLDER = "./owndata/traffic_label/test/"

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_300_vgg' in locals() else None
ssd_params = ssd_vgg_300.SSDNet.default_params._replace(
            num_classes=4)
ssd_net = ssd_vgg_300.SSDNet(ssd_params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.image("Image", image_4d))
    f_i = 0
    # for predict_map in predictions:
        # predict_map = predict_map[:, :, :, :, 1:]
        # predict_map = tf.reduce_max(predict_map, axis=4)
        # if f_i < 3:
            # predict_list = tf.split(predict_map, 6, axis=3)
            # anchor_index = 1
            # for anchor in predict_list:
                # summaries.add(tf.summary.image("predicte_map_%d_anchor%d" % (f_i,anchor_index), tf.cast(anchor, tf.float32)))
                # anchor_index += 1
        # else:
            # predict_map = tf.reduce_max(predict_map, axis=3)
            # predict_map = tf.expand_dims(predict_map, -1)
            # summaries.add(tf.summary.image("predicte_map_%d" % f_i, tf.cast(predict_map, tf.float32)))
        # f_i += 1
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


# Restore SSD model.
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/checkpoint'))
# if that checkpoint exists, restore from checkpoint
saver = tf.train.Saver()
summer_writer = tf.summary.FileWriter("./logs_test/", isess.graph)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(isess, ckpt.model_checkpoint_path)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.45, nms_threshold=0.45,
                  net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img, summary_op_str = isess.run([image_4d, predictions, localisations, bbox_img, summary_op],
                                                              feed_dict={img_input: img})


    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape,
        num_classes=4, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    summer_writer.add_summary(summary_op_str, 1)
    print rclasses, rscores, rbboxes
    return rclasses, rscores, rbboxes

def draw_results(img, rclasses, rscores, rbboxes, index):

    height, width, channels = img.shape[:3]
    for i in range(len(rclasses)):
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        cv2.putText(img, str(rclasses[i]) + ' ' +str(rscores[i]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    #cv2.imshow("predict", img)
    cv2.imwrite('./test_result/test_%d.jpg' % index, img)
def modify_xml(label_file_name, img, rclasses, rscores, rbboxes, index):

    label_file_path = LABEL_FOLDER + label_file_name
    tree = ET.parse(label_file_path)
    root = tree.getroot()
    # new_object = ET.Element('object')
    # root.append(new_object)
    # tree.write(label_file_path)
    height, width, channels = img.shape[:3]
    object_list = []
    for i in range(len(rclasses)):
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)
        new_object = ET.Element('object')
        name = ET.SubElement(new_object, 'name')
        name.text = 'Car'
        pose = ET.SubElement(new_object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(new_object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(new_object, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(new_object, 'bndbox')
        box_xmin = ET.SubElement(bndbox, 'xmin')
        box_xmin.text = str(xmin)
        box_ymin = ET.SubElement(bndbox, 'ymin')
        box_ymin.text = str(ymin)
        box_xmax = ET.SubElement(bndbox, 'xmax')
        box_xmax.text = str(xmax)
        box_ymax = ET.SubElement(bndbox, 'ymax')
        box_ymax.text = str(ymax)
        object_list.append(new_object)

    for obj in object_list:
        root.append(obj)
        # cv2.putText(img, str(rclasses[i]) + ' ' +str(rscores[i]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    tree.write(label_file_path, pretty_print=True)

path = LABEL_FOLDER 
label_file_names = sorted(os.listdir(path))
print label_file_names 
index = 1
for label_file_name in label_file_names:
    img_filename = label_file_name.split('.')[0]+'.png'
    print img_filename
    img = cv2.imread(IMAGE_FOLDER + img_filename)
    destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rclasses, rscores, rbboxes =  process_image(destRGB)
    modify_xml(label_file_name, img, rclasses, rscores, rbboxes, index)
    index += 1
