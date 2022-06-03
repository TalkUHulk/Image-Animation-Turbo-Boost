import torch
from modules.tpsmm import TPSMM
import yaml
from modules.keypoint_detector import KPDetector
import argparse
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("TPSMM onnx deploy")
    parser.add_argument(
        "--output-name-kp", type=str, default="kp_detector.onnx", help="output name of kp_detector models"
    )
    parser.add_argument(
        "--output-name-tpsmm", type=str, default="tpsmm.onnx", help="output name of tpsmm models"
    )
    parser.add_argument(
        "--input-kp", default="source", type=str, help="input node name of kp onnx model"
    )
    parser.add_argument(
        "--input-tpsmm", default=["kp_source", "source", "driving"], type=list,
        help="input node name of tpsmm onnx model"
    )
    parser.add_argument(
        "--output-kp", default="kp_source", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "--output-tpsmm", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--config",
        default="config/vox-256.yaml",
        type=str,
        help="yaml config file",
    )

    parser.add_argument("-c", "--ckpt", default="./checkpoints/vox.pth.tar", type=str, help="ckpt path")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")

    with open(args.config) as f:
        config = yaml.load(f)

    kp_detector = KPDetector(10)
    tpsmm_model = TPSMM(config)

    tpsmm_model.bg_predictor.load_state_dict(checkpoint["bg_predictor"])
    tpsmm_model.kp_detector.load_state_dict(checkpoint["kp_detector"])
    merge_dict = {}
    for key, value in checkpoint["dense_motion_network"].items():
        merge_dict[key] = value

    for key, value in checkpoint["inpainting_network"].items():
        merge_dict[key] = value

    tpsmm_model.dense_motion_inpainting_network.load_state_dict(merge_dict)

    kp_detector.load_state_dict(checkpoint["kp_detector"])
    kp_detector.eval()
    tpsmm_model.eval()

    dummy_source = torch.randn(args.batch_size, 3, 256, 256)
    dummy_driving = torch.randn(args.batch_size, 3, 256, 256)
    dummy_driving_kp = torch.randn(args.batch_size, 50, 2)

    torch.onnx.export(
        kp_detector,
        dummy_source,
        args.output_name_kp,
        input_names=[args.input_kp],
        output_names=[args.output_kp],
        dynamic_axes={args.input_kp: {0: 'batch'},
                      args.output_kp: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_kp))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_kp)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_kp)
        logger.info("generated simplified onnx model named {}".format(args.output_name_kp))

    torch.onnx.export(
        tpsmm_model,
        (dummy_driving_kp, dummy_source, dummy_driving),
        args.output_name_tpsmm,
        input_names=args.input_tpsmm,
        output_names=[args.output_tpsmm],
        dynamic_axes={args.input_tpsmm[0]: {0: 'batch'},
                      args.input_tpsmm[1]: {0: 'batch'},
                      args.input_tpsmm[2]: {0: 'batch'},
                      args.output_tpsmm: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_tpsmm))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_tpsmm)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_tpsmm)
        logger.info("generated simplified onnx model named {}".format(args.output_name_tpsmm))


if __name__ == "__main__":
    main()

