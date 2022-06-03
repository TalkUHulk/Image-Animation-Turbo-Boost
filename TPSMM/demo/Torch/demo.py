import time
from tqdm import tqdm
import torch
import numpy as np
import cv2
from skimage import img_as_ubyte
import yaml
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.tpsmm import TPSMM
from modules.keypoint_detector import KPDetector


def make_parser():
    parser = argparse.ArgumentParser("torch demo")
    parser.add_argument(
        "--source", type=str, help="input source image"
    )
    parser.add_argument(
        "--driving", type=str, help="input driving video"
    )
    parser.add_argument(
        "--output", default="./generated_torch.mp4", type=str, help="generated video path"
    )

    parser.add_argument(
        "-f",
        "--config",
        default="config/vox-256.yaml",
        type=str,
        help="yaml config file",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        type=str,
        help="device",
    )
    parser.add_argument("-c", "--ckpt", default="./checkpoints/vox.pth.tar", type=str, help="ckpt path")

    return parser


def reconstruction(kp_detector, tpsm_model, args):
    checkpoint = torch.load(args.ckpt, map_location=args.device)

    tpsm_model.bg_predictor.load_state_dict(checkpoint["bg_predictor"])
    tpsm_model.kp_detector.load_state_dict(checkpoint["kp_detector"])
    merge_dict = {}
    for key, value in checkpoint["dense_motion_network"].items():
        merge_dict[key] = value

    for key, value in checkpoint["inpainting_network"].items():
        merge_dict[key] = value

    tpsm_model.dense_motion_inpainting_network.load_state_dict(merge_dict)
    # tpsm_model.dense_motion_network.load_state_dict(checkpoint["dense_motion_network"])
    # tpsm_model.inpainting.load_state_dict(checkpoint["inpainting_network"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    kp_detector.eval()
    tpsm_model.eval()

    cap = cv2.VideoCapture(args.driving)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outvideo = cv2.VideoWriter(args.output, fourcc, 12, (256 * 3, 256), True)
    source = cv2.imread(args.source)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    source = cv2.resize(source, (256, 256))
    cv2_source = source.astype('float32') / 255
    source = torch.tensor(cv2_source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

    with torch.no_grad():
        if torch.cuda.is_available():
            source = source.cuda()
        kp_source = kp_detector(source)  # 1, 50, 2

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
        cnt = 0
        while True:
            cnt += 1

            ret, frame = cap.read()
            if not ret:
                break

            if cnt <= 50:
                continue
            if cnt == 70:
                break

            h, w, _ = frame.shape
            roi = frame[:, (w - h) // 2: (w + h) // 2, :]
            frame_face = cv2.resize(roi, (256, 256)) / 255
            frame_face_save = frame_face.copy()
            frame_face = torch.tensor(frame_face[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            if torch.cuda.is_available():
                frame_face = frame_face.cuda()

            driving = frame_face
            tic = time.time()
            out = tpsm_model(kp_source, source, driving)
            toc = time.time()
            im = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            joinedFrame = np.concatenate((cv2_source, im, frame_face_save), axis=1)

            outvideo.write(img_as_ubyte(joinedFrame))
            pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
            pbar.update(1)

    cap.release()
    outvideo.release()


def main():
    args = make_parser().parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f)

    kp_detector = KPDetector(10)
    tpsm_model = TPSMM(config)

    reconstruction(kp_detector, tpsm_model, args)


if __name__ == "__main__":
    main()

# python demo/Torch/test.py --source ../assets/source.png --driving ../assets/driving.mp4 --config ./config/vox-256.yaml --ckpt /Users/hulk/Documents/CodeZoo/Thin-Plate-Spline-Motion-Model-main/checkpoints/vox.pth.tar
