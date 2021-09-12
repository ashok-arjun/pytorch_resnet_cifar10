import argparse
import models

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet20')

parser.add_argument("--num-channels", nargs="+", type=int, default=[64,128,256,512])
parser.add_argument("--batch-norm", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--safe-region", type=str2bool, nargs='?', const=True, default=True)


args = parser.parse_args()

net = getattr(models, args.arch)(num_channels=args.num_channels, \
    batch_norm=args.batch_norm, safe_region=args.safe_region)

print(net)
