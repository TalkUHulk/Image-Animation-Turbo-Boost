from tqdm import tqdm
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
import multiprocessing
import onnxruntime
import argparse


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./generated_onnx.mp4", type=str, help="generated video path"
    )

    parser.add_argument("-c", "--onnx-file-tpsmm", type=str, help="tpsmm onnx model path")
    parser.add_argument("-k", "--onnx-file-kp", type=str, help="kp onnx model path")

    return parser


def main():
    args = make_parser().parse_args()

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    kp_detector = onnxruntime.InferenceSession(args.onnx_file_kp, sess_options, providers=['CPUExecutionProvider'])
    tpsm_model = onnxruntime.InferenceSession(args.onnx_file_tpsmm, sess_options, providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(args.driving)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outvideo = cv2.VideoWriter(args.output, fourcc, 12, (256 * 3, 256), True)
    source = cv2.imread(args.source)
    source = cv2.resize(source, (256, 256))
    cv2_source = source.astype('float32') / 255
    source = cv2.cvtColor(cv2_source, cv2.COLOR_BGR2RGB)
    source = np.transpose(source[np.newaxis].astype(np.float32), (0, 3, 1, 2))

    ort_inputs = {kp_detector.get_inputs()[0].name: source}
    kp_source = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]  # 1, 50, 2

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_face = cv2.resize(frame, (256, 256)) / 255
        frame_face_save = frame_face.copy()
        frame_face = np.transpose(frame_face[np.newaxis].astype(np.float32), (0, 3, 1, 2))

        driving = frame_face
        tic = time.time()
        ort_inputs = {tpsm_model.get_inputs()[0].name: kp_source,
                      tpsm_model.get_inputs()[1].name: source,
                      tpsm_model.get_inputs()[2].name: driving}
        out = tpsm_model.run([tpsm_model.get_outputs()[0].name], ort_inputs)[0]
        toc = time.time()
        im = np.transpose(out, [0, 2, 3, 1])[0]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        joinedFrame = np.concatenate((cv2_source, im, frame_face_save), axis=1)

        outvideo.write(img_as_ubyte(joinedFrame))

        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)

    cap.release()
    outvideo.release()


if __name__ == "__main__":
    main()
