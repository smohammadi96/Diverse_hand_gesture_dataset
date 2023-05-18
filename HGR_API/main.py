from __future__ import print_function
import base64
import json

import argparse
import grpc
from itertools import islice
import numpy as np
import requests
import cv2
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from flask import Flask, request
from flask_cors import CORS

import core.utils as utils
from shenasa_utils.validation import Validation


app = Flask(__name__)
CORS(app)
validation = Validation()


def extract_hand_gestures(image):
    """

    :param image: A dictionary consists of numpy array.
    :returns: list of class names of detected gestures, list of bboxes of gestures, number of detected gestures.
    """
    for key, val in image.items():
        original_image = cv2.cvtColor(val[0], cv2.COLOR_BGR2RGB)
    
    image_data = cv2.resize(original_image, (416, 416))
    img  = image_data / 255.
    img = img.reshape((1, 416, 416, 3))

    # Wrap bitstring in JSON and POST to server, then wait for response
    '''
    instance = {"input_1": img.tolist()}
    data = json.dumps({"inputs": instance})
    header = {
            'content-type': 'application/json'
    }
    '''
    # Create stub
    server_url = 'gesture_ex:8500'
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'serving_model'
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(img.astype(np.float32)))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    outputs_tensor_proto = result.outputs["tf_op_layer_concat_18"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    response = tf.constant(outputs_tensor_proto.float_val, shape=shape)
    #post to Rest API
    '''
    json_response = requests.post("http://localhost:9000/v1/models/serving_model:predict", data=data, headers=header)
    '''
    boxes = response[:, :, 0:4]
    pred_conf = response[:, :, 4:]
        
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    class_name, bbox = utils.draw_bbox(original_image, pred_bbox)
    
    return  class_name, bbox, len(class_name)

# Send image to server, receive inferred output
@app.route('/hand_gesture', methods=['POST'])
def hand_gesture():
    if validation.validate(request):

        image = validation.get_file()
        with open('readme.txt', 'w') as f:
            f.write(str(image))
        class_name, bbox, n_gestures = extract_hand_gestures(image)
        data = []
        for i in range(1,int(n_gestures)+1):    
            result = {}
            result['gesture'] = class_name[i-1]
            result['x1'] = int(bbox[i-1][0][0])
            result['y1'] = int(bbox[i-1][0][1])
            result['x2'] = int(bbox[i-1][1][0])
            result['y2'] = int(bbox[i-1][1][1])
            data.append(result)
        
        if len(bbox) > 0:
            
            return {
                    'code': 'success',
                    'message': 'detection completed.',
                    'data': json.dumps(data),
                    'status': True
            }
        else:

            return {
                    'code': 'NoGestureFound',
                    'message': 'detection completed.',
                    'data': [],
                    'status': False
            }


    else:

        error = validation.get_error_code()
        
        return {
                'code': error,
                'message': 'detection failed.',
                'data': [],
                'status': False
        }, 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6004)

