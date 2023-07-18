from flask import Flask, jsonify, request, render_template, redirect
import os
# import onnx
from PIL import Image
import os
# import cv2
import time
import glob
import tensorflow as tf
import torchvision
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import argparse
import onnx
import json
from onnx2torch import convert
import pathlib

import sqlite3
print(torch.__file__)

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "C:\\Users\\tecks\\OneDrive\\Desktop\\podor\\uploaded_images"
model_path_rcnn = "C:\\Users\\tecks\\OneDrive\\Desktop\\podor\\maskrcnn_v2\\"
model_path_deeplab = 'C:\\Users\\Priya\\Downloads\\deeplabv3\\Model\\'

width = 1024
height = 1024
threshold = 0.5
# maskedrcnn_output_folder = "C:\\Users\\Priya\\OneDrive\\Documents\\SMU\\Study\\Semester 4\\Deep learning for visual recognition\\Project\\podor\\maskedrcnn_predictions\\"
maskedrcnn_output_folder = "C:\\Users\\tecks\\OneDrive\\Desktop\\podor\\maskrcnn_v2"
# deeplabv3_output_folder = "C:\\Users\\Priya\\OneDrive\\Documents\\SMU\\Study\\Semester 4\\Deep learning for visual recognition\\Project\\podor\\deeplabv3_predictions\\"
deeplabv3_output_folder = "C:\\Users\\tecks\\OneDrive\\Desktop\\podor\\deeplabv3_predictions"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Route to upload image
@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print('uploaded image', image)
            img_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(img_path)
            return render_template("podor-home.html", img_path = img_path, img_filename = image.filename)
    return render_template("podor-home.html")


@app.route("/pothole-prediction", methods=["POST"])
def pothole_prediction():
    if request.method == "POST":
        formdata = request.form
        img_path=formdata.get("img_path")
        img_filename=formdata.get("img_filename")
        model_selected=formdata.get("model")
        data = []
        if model_selected=='deeplabv3':
            data=get_deeplabv3(img_path,img_filename)
        else:
            data=get_maskedrcnn(img_path,img_filename)
        print(jsonify(data))
        print(json.loads(jsonify(data).get_data().decode("utf-8")))
    # return jsonify(data)
    return render_template("podor-home.html", predictions = json.loads(jsonify(data).get_data()))


def get_maskedrcnn(img_path,img_name):
    # color_map, detect_fn = get_maskrcnn_model()
    print('color map', color_map, 'detect_fn', detect_fn)

    masks, bboxes, classes, scores = get_prediction(img_path, color_map, detect_fn)
    insert_record((str(img_path), str(masks), str(bboxes), str(classes), str(scores)))
    output_img_path=''
    if len(masks) != 0:
        output_img_path = maskedrcnn_output_folder + img_name
    create_table()
    area_bboxes=get_boundingingboxarea(bboxes)
    pothole_count=len(scores)
    severity=""
    if len(area_bboxes)>0:
        if max(area_bboxes)>=0.2:
            severity="high"
        if max(area_bboxes)>0.17 and max(area_bboxes)<0.2:
            severity="medium"
        if max(area_bboxes)<=0.17:
            severity="low"
    data = {"masks": str(masks), "img_path": img_path, "masked_image_path": output_img_path,
            "boundaryboxes": str(bboxes), "classes": str(classes), "scores": str(scores),"area_bboxes":str(area_bboxes),
            "number_of_potholes":str(pothole_count),"severity":str(severity)}
    return data



def get_boundingingboxarea(bboxes):
    area_bboxes=[]
    for bbox in bboxes:
        area=(bbox[3]-bbox[1]) * (bbox[2]- bbox[0])
        area_bboxes.append(area)
    return area_bboxes



def create_table():
    conn = sqlite3.connect('my_database.sqlite')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS PODOR
                 (ID INTEGER PRIMARY KEY     AUTOINCREMENT ,
                 IMG_PATH       CHAR(50)     NOT NULL,
                 MASKS           TEXT,
                 BBOXES           TEXT,
                 CLASSES          TEXT,
                 SCORES          TEXT
                 );''')
    print("Created database successfully")
    cursor.close()

def insert_record(data_tuple):
    conn = sqlite3.connect('my_database.sqlite')
    cursor = conn.cursor()

    sqlite_insert_query = """ INSERT INTO PODOR
                                      (img_path, masks, bboxes, classes, scores) VALUES (?, ?, ?, ?, ?)"""
    cursor.execute(sqlite_insert_query, data_tuple)
    print("Inserted tuple", data_tuple)
    conn.commit()
    cursor.close()


'''masked RCNN'''


def apply_mask(image, mask, colors, alpha=0.5):
    """Apply the given mask to the image.

    Args:
      image: original image array.
      mask: predict mask array of image.
      colors: color to apply for mask.
      alpha: transparency of mask.

    Returns:
      array of image with mask overlay
    """
    for color in range(3):
        image[:, :, color] = np.where(
            mask == 1,
            image[:, :, color] * (1 - alpha) + alpha * colors[color],
            image[:, :, color],
        )
    return image


'''masked RCNN'''


def get_prediction(image_path, color_map, detect_fn):

    print("MaskedRCNN Prediction for {}...".format(image_path))

    ## Returned original_shape is in the format of width, height
    image_resized, origi_shape = load_image_into_numpy_array(image_path, int(height), int(width))
    # input_tensor, origi_shape = load_image_into_numpy_array(each_image, int(height), int(width))

    ## The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_resized, dtype='uint8')

    ## The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    ## Feed image into model
    trained_model = detect_fn.signatures["serving_default"]
    detections = trained_model(input_tensor)
    # detections = model.run(input_tensor[0])

    ## Process predictions
    num_detections = int(detections.pop("num_detections"))

    need_detection_key = [
        "detection_classes",
        "detection_boxes",
        "detection_masks",
        "detection_scores",
    ]

    predictions = {key: detections[key][0, :num_detections].numpy() for key in need_detection_key}

    ## Filter out predictions below threshold
    predictions["num_detections"] = num_detections
    indexes = np.where(predictions["detection_scores"] > float(threshold))

    if "detection_masks" in predictions:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            tf.convert_to_tensor(predictions["detection_masks"]),
            predictions["detection_boxes"],
            origi_shape[0],
            origi_shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        predictions["detection_masks_reframed"] = detection_masks_reframed.numpy()

    ## Extract predictions
    masks = predictions["detection_masks_reframed"][indexes]
    bboxes = predictions["detection_boxes"][indexes]
    classes= predictions["detection_classes"][indexes].astype(np.int64)
    scores = predictions["detection_scores"][indexes]
    print('Prediction results - masks:',masks, ', bboxes:', bboxes, ', classes:', classes, ', scores:', scores)
    # ## Draw Predictions
    image_origi = Image.fromarray(image_resized).resize(
        (origi_shape[1], origi_shape[0])
    )
    image_origi = np.array(image_origi)

    if len(masks) != 0:
        for idx, each_bbox in enumerate(bboxes):
            color = color_map.get(classes[idx] - 1)
            masked_image = apply_mask(image_origi, masks[idx], color)  # Segmentation mask

        ## Save predicted image
        filename = os.path.basename(image_path)
        image_predict = Image.fromarray(masked_image)
        print(image_predict)
        image_predict.save(os.path.join(maskedrcnn_output_folder, filename))
    else:
        print('No potholes detected')

    return masks, bboxes, classes, scores


'''masked RCNN'''


def load_label_map(label_map_path):
    """Reads label map in the format of .pbtxt and parse into dictionary

    Args:
      label_map_path: the file path to the label_map

    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {"id": label_index, "name": label_name}

    return label_map


'''masked RCNN'''


def load_image_into_numpy_array(path, height, width):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape

    image_resized = image.resize((height, width))
    # image = image.resize((height, width))
    # image = np.asarray(image, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
    # image = image.transpose(0, 1, 4, 2, 3)
    # image = image.reshape(1, 3, height, width)
    return np.array(image_resized), (image_shape[0], image_shape[1])


'''masked RCNN'''


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width, resize_method="bilinear"):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
      box_masks: A tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.
      resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
        'bilinear' is only respected if box_masks is a float.

    Returns:
      A tensor of size [num_masks, image_height, image_width] with the same dtype
      as `box_masks`.
    """
    resize_method = "nearest" if box_masks.dtype == tf.uint8 else resize_method

    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            denom = max_corner - min_corner
            # Prevent a divide by zero.
            denom = tf.math.maximum(denom, 1e-4)
            transformed_boxes = (boxes - min_corner) / denom
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat([tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], 1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

        resized_crops = tf.image.crop_and_resize(
            box_masks_expanded,
            reverse_boxes,
            tf.range(num_boxes),
            [image_height, image_width],
            method=resize_method,
            extrapolation_value=0,
        )
        return tf.cast(resized_crops, box_masks.dtype)

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype),
    )
    return tf.squeeze(image_masks, axis=3)



'''masked RCNN'''


def get_maskrcnn_model():
    # Loading tensorflow model
    print('Loading model from ', model_path_rcnn)
    detect_fn = tf.saved_model.load(model_path_rcnn + 'saved_model')
    print('Loading model complete')
    # Loading labels
    label_fp = model_path_rcnn + 'label_map.pbtxt'
    category_index = load_label_map(label_fp)
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [int(i) for i in np.random.choice(range(256), size=3)]
    return color_map, detect_fn



'''deeplabv3'''


def get_deeplabv3(img_path,img_name):
    print("Deeplabv3 Prediction for {}...".format(img_path))
    # loading the labels
    # label_fp = "/content/drive/MyDrive/CS604-Project/deeplabv3/Model/label_map.pbtxt"
    label_fp = model_path_deeplab + 'label_map.pbtxt'
    # print("Deeplabv3 label map")

    category_index = load_label_map(label_fp)
    # print("Deeplabv3 label map complete")
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [int(i) for i in np.random.choice(range(256), size=3)]
    # print("Deeplabv3 color map complete")
    label_map = [[0, 0, 0]]
    label_map.append(color_map[0])

    image = Image.open(img_path).convert('RGB')
    # do forward pass and get the output dictionary
    print("Deeplabv3 get segment labels")
    outputs = get_segment_labels(image, torch_model, device)
    print("Deeplabv3 get segment labels complete")
    # get the data from the `out` key
    outputs_2 = outputs[0]
    segmented_image = draw_segmentation_map(outputs_2, label_map)
    final_image = image_overlay(image, segmented_image)
    # save_name = f"test2_done"
    # image_predict = Image.fromarray(masked_image)

    output_img_path = os.path.join(deeplabv3_output_folder, img_name)
    extension = os.path.splitext(img_name)[1][1:].strip().lower()

    print('Saving image at ', output_img_path, 'extension=', extension)
    image_predict = Image.fromarray(final_image)
    print(image_predict)

    image_predict.save(output_img_path, extension)

    data = {"img_path" : img_path, "masked_image_path":output_img_path }
    return data


'''deeplabv3'''


def get_segment_labels(image, model, device):
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    # transform the image to tensor and load into computation device
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image)
    return outputs


'''deeplabv3'''


def draw_segmentation_map(outputs, label_map):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map


'''deeplabv3'''


def image_overlay(image, segmented_image):
    alpha = 1.0 # transparency for the original image
    beta = 0.8 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image


'''deeplabv3'''


def load_label_map(label_map_path):
    """Reads label map in the format of .pbtxt and parse into dictionary

    Args:
      label_map_path: the file path to the label_map

    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {"id": label_index, "name": label_name}

    return label_map





print('Loading MaskedRCNN model...')
color_map, detect_fn = get_maskrcnn_model()
print('Loading MaskedRCNN model complete')

print('Loading Deeplabv3 model...')
# onnx_model = onnx.load(model_path_deeplab + 'model.onnx')
# torch_model = convert(onnx_model)
# torch_model.eval().to(device)
print('Loading Deeplabv3 model complete')


if __name__ == '__main__':
    app.run(debug=True)




