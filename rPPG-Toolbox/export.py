import argparse
import sys
import os
import torch
from collections import OrderedDict
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./UBFC-rPPG_EfficientPhys.pth', help='weights path')
    parser.add_argument('--export_dir', default=ROOT / 'ExportModels', help='dir to save exported weights')
    opt = parser.parse_args()

    export_weight = OrderedDict()
    for k, v in torch.load(opt.weights).items():
        name = k[7:]  # remove "module."
        export_weight[name] = v

    if not os.path.exists(opt.export_dir):
        os.makedirs(opt.export_dir)
    model_path = os.path.join(opt.export_dir, os.path.basename(opt.weights))
    torch.save(export_weight, model_path)
    print('Saved Model Path: ', model_path)
