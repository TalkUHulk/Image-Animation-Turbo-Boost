import yaml
import torch
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
import argparse
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("FOMM onnx deploy")
    parser.add_argument(
        "--output-name-kp", type=str, default="kp_detector.onnx", help="output name of kp_detector models"
    )
    parser.add_argument(
        "--output-name-fomm", type=str, default="fomm.onnx", help="output name of fomm models"
    )
    parser.add_argument(
        "--input-kp", default="driving_frame", type=str, help="input node name of kp onnx model"
    )
    parser.add_argument(
        "--input-fomm", default=["source", "driving_value", "driving_jacobian", "source_value", "source_jacobian"],
        type=list,
        help="input node name of tpsmm onnx model"
    )
    parser.add_argument(
        "--output-kp", default=["driving_value", "driving_jacobian"], type=list, help="output node name of onnx model"
    )
    parser.add_argument(
        "--output-fomm", default="output", type=str, help="output node name of onnx model"
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
        default="config/vox-adv-256.yaml",
        type=str,
        help="yaml config file",
    )

    parser.add_argument("-c", "--ckpt", default="./checkpoints/vox-adv-cpk.pth.tar", type=str, help="ckpt path")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()

    dummy_input = torch.randn(args.batch_size, 3, 256, 256)
    torch.onnx.export(
        kp_detector,
        dummy_input,
        args.output_name_kp,
        input_names=[args.input_kp],
        output_names=args.output_kp,
        dynamic_axes={args.input_kp: {0: 'batch'},
                      args.output_kp[0]: {0: 'batch'},
                      args.output_kp[1]: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
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


    source = torch.randn(1, 3, 256, 256)
    driving_value = torch.randn(1, 10, 2)
    driving_jacobian = torch.randn(1, 10, 2, 2)

    source_value = torch.randn(1, 10, 2)
    source_jacobian = torch.randn(1, 10, 2, 2)

    torch.onnx.export(
        generator,
        (source, driving_value, driving_jacobian, source_value, source_jacobian),
        args.output_name_fomm,
        input_names=args.input_fomm,
        output_names=[args.output_fomm],
        dynamic_axes={args.input_fomm[0]: {0: 'batch'},
                      args.input_fomm[1]: {0: 'batch'},
                      args.input_fomm[2]: {0: 'batch'},
                      args.input_fomm[3]: {0: 'batch'},
                      args.input_fomm[4]: {0: 'batch'},
                      args.output_fomm: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset
    )
    logger.info("generated onnx model named {}".format(args.output_name_fomm))
    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(args.output_name_fomm)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_fomm)
        logger.info("generated simplified onnx model named {}".format(args.output_name_fomm))


if __name__ == "__main__":
    main()

# python export_onnx.py --output-name-kp kp_detector.onnx --output-name-fomm fomm.onnx --config config/vox-adv-256.yaml --ckpt ./checkpoints/vox-adv-cpk.pth.tar