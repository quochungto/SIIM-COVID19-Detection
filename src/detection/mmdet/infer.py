# stdlib
import os
import argparse

# numlib
import pandas as pd

from global_config import Cfg
from func import allocate_files, get_image_sub

# modelname - fold - input size - epoch
# NOTE: pickle file output has the format of VOC [x_min y_min x_max y_max]
def infer(ck_path, image_size, cfg_path=Cfg.cfg_path, save_dir='../result/mmdet/submit', mode='remote'):

    ck_name = os.path.split(ck_path)[-1].split('.')[0]
    save_dir = os.path.join(save_dir, mode, ck_name)
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f'results.pkl')

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
        os.system(infer_command)

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
        --out {"../../../" + pkl_path} \
        --eval bbox'
        os.system(infer_command)

    os.chdir('../../..')
        
    df_valid.to_csv(os.path.join(save_dir, 'valid.csv'), index=False)

    df_sub = get_image_sub(pkl_path, df_valid, image_size)
    df_sub.to_csv(os.path.join(save_dir, f'results.csv'), index=False)

    print('\n')

    return df_sub

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck-paths', type=str, nargs='+')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--conf-thr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Cfg.seed)
    
    opt = parse_opt()
    
    ck_paths, image_size, batch_size, \
    iou, conf, mode, device = \
    opt.ck_paths, opt.image_size, opt.batch_size, \
    opt.iou_thr, opt.conf_thr, opt.mode, opt.device

    for ck_path in opt.ck_paths:
	infer(ck_path, Cfg.image_size, mode=opt.mode)

    #if WORKING_BASE == '/content':
#	    save_in_drive('/content/submit', prefix='mmdet_vfnet_submit')

if __name__ == '__main__':
    main()
