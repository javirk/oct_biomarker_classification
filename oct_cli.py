import argparse
from oct import OCTMultiLabelDetector
from pathlib import Path

def run_training(args):
    det = OCTMultiLabelDetector(args)
    det.load_datasets()
    det.load_model()
    det.train()

def infer(args):
    det = OCTMultiLabelDetector(args)
    det.load_datasets()
    det.load_model()
    det.infer(det.model, det.tb_path / 'inference')

parser = argparse.ArgumentParser(
    prog='OCT Detector', 
    epilog="See '<command> --help' to read about a specific sub-command.",
    fromfile_prefix_chars='@'
)
parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

# Common arguments to train and infer subparsers
parent_parser = argparse.ArgumentParser(add_help=False)                                 
parent_parser.add_argument('--test-data', action="store", type=int, required=True)
parent_parser.add_argument('--classes', action="store", type=int, required=True)
parent_parser.add_argument('--resize', action="store", type=int)
parent_parser.add_argument('--test-target', action="store", type=str, required=True)
parent_parser.add_argument('--model-name', action="store", type=str, required=True)
parent_parser.add_argument('--no-ubelix', action="store_false", dest="ubelix")
parent_parser.add_argument('--suffix', action="store", type=str, default="")

subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

train_parser = subparsers.add_parser('train', help='Train a model', parents=[parent_parser])
train_parser.add_argument("--ensemble", action="store_true")
train_parser.add_argument("--model-paths", action="extend", nargs="+", type=str)
train_parser.add_argument("--batch-size", action="store", type=int, required=True)
train_parser.add_argument("--epochs", action="store", type=int, required=True)
train_parser.add_argument("--learning-rate", action="store", type=float, required=True)
train_parser.add_argument("--writing-per-epoch", action="store", type=int, default=10)
train_parser.add_argument('--train-data', action="store", type=int, required=True)
train_parser.add_argument('--train-target', action="store", type=str, required=True)
train_parser.add_argument('--no-pretrain', action="store_false", dest="pretrained")
train_parser.add_argument('--restart-from', action="store", type=str)
# train_parser.set_defaults(func=run_command)

infer_parser = subparsers.add_parser('infer', help='Run inference with a trained model', parents=[parent_parser])
infer_parser.add_argument('--model-path', default="store", type=str, required=True)
infer_parser.add_argument("--batch-size", action="store", type=int, default=32)
infer_parser.add_argument('--baseline', action='store_true')
infer_parser.add_argument('--no-baseline', dest='baseline', action='store_false')
infer_parser.set_defaults(baseline=True)
# infer_parser.set_defaults(func=run_command)

args = parser.parse_args()

if args.command == 'train':
    if args.ensemble and args.model_paths is None:
        parser.error("--ensemble requires --model-paths to be specified.")
    elif not args.ensemble and args.model_paths is not None:
        parser.error("Cannot specify --model-paths without --ensemble.")

if args.command == 'train':
    run_training(args)
elif args.command == 'infer':
    infer(args)