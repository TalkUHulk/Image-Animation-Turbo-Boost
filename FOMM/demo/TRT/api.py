from flask import request, jsonify
from flask import Blueprint

fom_bp = Blueprint("fom", __name__)

import time
from trt_model import TrtModels
import base64
import cv2
import numpy as np
from utils import normalize_kp
import imageio
import pycuda
import torch

source = np.random.random([1, 3, 256, 256]).astype(np.float32)
generator = TrtModels("weights/generator_sim.trt", pycuda.autoinit.context)
kp_detector = TrtModels("weights/kp_detector_sim.trt", pycuda.autoinit.context)
kp_source_value = np.random.random([1, 10, 2]).astype(np.float32)
kp_source_jacobian = np.random.random([1, 10, 2, 2]).astype(np.float32)
kp_driving_initial_value = np.random.random([1, 10, 2]).astype(np.float32)
kp_driving_initial_jacobian = np.random.random([1, 10, 2, 2]).astype(np.float32)


@fom_bp.route("/fom", methods=["POST"], endpoint="fom")
def FOM():
    tic = time.time()
    ret = {}
    try:
        global kp_source_value
        global kp_source_jacobian
        global kp_driving_initial_value
        global kp_driving_initial_jacobian
        global source

        content = request.json
        image_raw = content["image_raw"]
        is_source = content["is_source"]
        is_initial = content["is_initial"]

        image = base64.b64decode(image_raw)

        if is_source:
            source_image = imageio.imread(image)[..., :3] / 255
            x_input = np.transpose(source_image[np.newaxis].astype(np.float32), (0, 3, 1, 2))
            source = x_input
            kp_value, kp_jacobian = kp_detector(x_input)
            kp_source_value = np.resize(kp_value, [1, 10, 2])
            kp_source_jacobian = np.resize(kp_jacobian, [1, 10, 2, 2])
        else:
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            x_input = np.transpose(image[np.newaxis].astype(np.float32), (0, 3, 1, 2)) / 255
            if is_initial:
                kp_value, kp_jacobian = kp_detector(x_input)
                kp_driving_initial_value = np.resize(kp_value, [1, 10, 2])
                kp_driving_initial_jacobian = np.resize(kp_jacobian, [1, 10, 2, 2])
                kp_driving_value = kp_driving_initial_value
                kp_driving_jacobian = kp_driving_initial_jacobian
            else:
                kp_value, kp_jacobian = kp_detector(x_input)
                kp_driving_value = np.resize(kp_value, [1, 10, 2])
                kp_driving_jacobian = np.resize(kp_jacobian, [1, 10, 2, 2])

            kp_norm_value, kp_norm_jacobian = normalize_kp(kp_source_value,
                                                           kp_source_jacobian,
                                                           kp_driving_value,
                                                           kp_driving_jacobian,
                                                           kp_driving_initial_value,
                                                           kp_driving_initial_jacobian)
            output = generator(source,
                               kp_norm_value,
                               kp_norm_jacobian,
                               kp_source_value,
                               kp_source_jacobian)[0]

            output = np.resize(output, (1, 3, 256, 256))
            im_out = np.transpose(output, [0, 2, 3, 1])[0] * 255
            im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
            image = cv2.imencode('.jpg', im_out)[1]
            image_code = base64.b64encode(image).decode('utf8')
            ret["image_raw"] = image_code
        ret["code"] = 0
        ret["msg"] = "succeed"
    except Exception as e:
        print(e)
        ret["code"] = -1
        ret["msg"] = e.__str__()
    finally:
        ret["time"] = round(time.time() - tic, 5)
        return jsonify(ret)
