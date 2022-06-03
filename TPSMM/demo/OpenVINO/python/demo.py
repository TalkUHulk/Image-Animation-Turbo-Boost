import time
from openvino.inference_engine import IECore
from tqdm import tqdm
import numpy as np
import cv2
from skimage import img_as_ubyte
import argparse


def make_parser():
    parser = argparse.ArgumentParser("openvino demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./generated_opv.mp4", type=str, help="generated video path"
    )

    parser.add_argument("--xml-tpsmm", type=str, help="tpsmm xml path")
    parser.add_argument("--xml-kp", type=str, help="kp xml path")

    parser.add_argument("--bin-tpsmm", type=str, help="tpsmm bin path")
    parser.add_argument("--bin-kp", type=str, help="kp onnx bin ")

    return parser


def main():
    args = make_parser().parse_args()

    ie = IECore()
    kp_detector_net = ie.read_network(model=args.xml_kp, weights=args.bin_kp)
    kp_detector_model = ie.load_network(kp_detector_net, "CPU")

    net = ie.read_network(model=args.xml_tpsmm, weights=args.bin_tpsmm)
    tpsm_model = ie.load_network(net, "CPU")

    cap = cv2.VideoCapture(args.driving)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outvideo = cv2.VideoWriter(args.output, fourcc, 12, (256 * 3, 256), True)
    source = cv2.imread(args.source)
    source = cv2.resize(source, (256, 256))
    cv2_source = source.astype('float32') / 255
    source = cv2.cvtColor(cv2_source, cv2.COLOR_BGR2RGB)
    source = np.transpose(source[np.newaxis].astype(np.float32), (0, 3, 1, 2))

    res = kp_detector_model.infer(inputs={"source": source})
    kp_source = res["kp_source"]

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
        res = tpsm_model.infer(inputs={"driving": driving,
                                       "source": source,
                                       "kp_source": kp_source})
        toc = time.time()
        output = res["output"]

        im = np.transpose(output, [0, 2, 3, 1])[0]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        joinedFrame = np.concatenate((cv2_source, im, frame_face_save), axis=1)
        outvideo.write(img_as_ubyte(joinedFrame))
        pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
        pbar.update(1)

    cap.release()
    outvideo.release()


if __name__ == "__main__":
    main()

