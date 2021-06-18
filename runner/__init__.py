import os
import os.path as osp
import torch
import yaml

from runner.args import get_args
from utils.alarm import get_alarm


args = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"

if args.gpu < 0:
    args.loc = 'local'

device = torch.device('cuda' if ((args.gpu >= 0) & torch.cuda.is_available()) else 'cpu')
args.device = device

with open("runner/paths.yaml") as f:
    _paths = yaml.load(f, Loader=yaml.FullLoader)
    paths = _paths[args.loc]
    datasets_path = paths["datasets_path"]
    evalsets_path = paths["evalsets_path"]
    results_path = paths["results_path"]

alarm = get_alarm(args.alarm, args.channel)
