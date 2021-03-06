import sys
sys.path.append('.')

# stdlib
import os
import argparse

# custom
from global_config import Cfg
from utils.torch_common import seed_everything, memory_cleanup
from func import allocate_files

# mmdetection
import mmdet
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed

def train(folds, epochs=25, cfg_path=Cfg.cfg_path):
    for fold in folds:
        memory_cleanup()

        print(f'---- Training fold {fold} ----')
        allocate_files(fold, is_train=True)

        cfg = Config.fromfile(cfg_path)
        cfg.work_dir = './exps/fold%d'%fold
        cfg.total_epochs = epochs
        cfg.runner['max_epochs'] = epochs
        cfg.data = cfg.data[0]
        datasets = build_dataset([cfg.data.train])
        model = build_detector(cfg.model)
        train_detector(model, datasets, cfg, validate=True, distributed=False)
        print('\n')
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='vfnetr50')
    parser.add_argument('--folds', type=str, nargs='+', \
            default=[0, 1, 2, 3, 4], help='folds for training, i.e. 0 or 0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Cfg.seed)
    opt = parse_opt()
    assert opt.weight in ['vfnetr50', 'vfnetr101'], 'invalid value for --weight'
    if opt.weight = 'vfnetr50':
        cfg_path = default=os.path.abspath('./detection/mmdet/model_configs/vfnetr50_cfg.py'))
    else:
        cfg_path = default=os.path.abspath('./detection/mmdet/model_configs/vfnetr101_cfg.py'))
    train(opt.folds, epochs=opt.epochs, cfg_path=cfg_path)

if __name__ == '__main__':
    main()
