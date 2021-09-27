# stdlib
import os
import argparse

# numlib
import pandas as pd

from global_config import Cfg
from func import allocate_files, get_image_sub

# modelname - fold - input size - epoch
# NOTE: pickle file output has the format of VOC [x_min y_min x_max y_max]
def infer(ck_path, cfg_path, image_size=Cfg.image_size, save_dir='../result/mmdet/submit', mode='remote'):

    ck_name = os.path.split(ck_path)[-1].split('.')[0]
    save_dir = os.path.join(save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    save_path = increment_path(os.path.abspath(os.path.join(save_dir, ck_name))) + '.csv'
    pkl_path = increment_path(os.path.abspath(os.path.join(save_dir, ck_name))) + '.pkl'

    print(f'Running {ck_name} - {mode}')
    print('\n')
    
    if mode == 'remote':
        df_valid = make_fold('test')
        allocate_files(fold=None, is_train=False)

        os.chdir('./detection/mmdet/mmdetection')

        infer_command = f'python \
        ./tools/test.py \
        {"../../../" + cfg_path} \
        {"../../../" + ck_path} \
        --out {pkl_path}'

    elif mode == 'local':
        fold = ck_name.split('_')[1]
        fold = int(fold.replace('fold', ''))

        _, df_valid = make_fold('train-%s'%fold)
        allocate_files(fold=fold, is_train=False)
        
        os.chdir(os.path.join(WORKING_BASE, 'mmdetection'))
        
        infer_command = 'python \
        ./tools/test.py \
        {"../../../" + cfg_path} \
        {"../../../" + ck_path} \
        --out {pkl_path} \
        --eval bbox'

    os.system(infer_command)
    os.chdir('../../..')
        
    #df_valid.to_csv(os.path.join(save_dir, 'valid.csv'), index=False)

    df_sub = get_image_sub(pkl_path, df_valid, image_size)
    #df_sub.to_csv(os.path.join(save_dir, f'results.csv'), index=False)
    df_sub.to_csv(save_path, index=False)

    print('\n')

    return df_sub

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck-paths', type=str, nargs='+')
    parser.add_argument('--cfg-path', type=str, \
            default=os.path.abspath('./detection/mmdet/model_configs/vfnetr50_cfg.py'))
    #parser.add_argument('--image-size', type=int, default=512)
    #parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--conf-thr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Cfg.seed)
    opt = parse_opt()
    replacement = ''
    with open(opt.cfg_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if 'iou_threshold' in line:
                line = f"       nms=dict(type='nms', iou_threshold={opt.iou_thr}),"
            elif 'score_thr' in line:
                line = f"       score_thr={opt.conf_thr},"
            replacement += replacement + line + '\n'
    with open(opt.cfg_path. 'w') as f:
        f.write(replacement)

    for ck_path in opt.ck_paths:
	infer(opt.ck_path, opt.cfg_path, mode=opt.mode)

if __name__ == '__main__':
    main()
