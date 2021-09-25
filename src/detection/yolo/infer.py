import sys
sys.path.append('.')

# stdlib
import os
import shutil
from glob import glob
from tqdm.auto import tqdm
import re
import time
import argparse

from global_config import Config
from utils.file import Logger
from utils.metrics import map_2cls
from utils.torch_common import seed_everything, memory_cleanup
from func import make_fold, allocate_files, get_image_sub

def yolo_infer(ck_path, image_size=512,
               batch_size=16,
               iou_thresh=0.5,
               conf_thresh=0.001,
               mode='remote',
               save_dir='../result/yolo/submit',
               fold_path=None,
               duplicate_path=None,
               device=0):
    t0 = time.time()
    memory_cleanup()
    yolo_ver = 'yolov5' if 'yolov5' in ck_path else 'yolotrs' if 'yolotrs' in ck_path else None
    assert yolo_ver is not None
    
    t = time.strftime('%Y%m%d_%H%M%S')
    fold = -1
    for s in ck_path.split('/'):
        for ss in s.split('.'):
            for sss in ss.split('_'):
                if len(sss) == 5 and 'fold' in sss:
                    fold = int(sss.replace('fold',''))
                    break
                if fold != -1: break

    assert fold > -1, 'checkpoint path is not in correct structure'
    
    ck_name = 'fold%d'%fold
    sname = re.sub('[^\w_-]', '', '%d_iou%.2f_conf%.4f'%(image_size, iou_thresh, conf_thresh))
    save_dir = os.path.join(save_dir, mode, sname, ck_name, t)
    os.makedirs(save_dir, exist_ok=True)

    #----logging
    log = Logger()
    log.open('../logging/yolo_valid.txt', mode='a')
    log.write(f'infer {ck_name} - fold {fold} - {sname} - {mode}\n'.upper())
    log.write(t+'\n')
    log.write(ck_path+'\n')
    log.write('mode=%s,fold=%d,batch_size=%d,image_size=%d,iou=%.4f,conf=%.4f\n'\
              %(mode,fold,batch_size,image_size,iou_thresh,conf_thresh))
    #----
    
    if mode == 'local':
        _, df_valid = make_fold('train-%d'%fold, Config.csv_path, fold_path, duplicate_path)
        allocate_files(fold, csv_path=Config.csv_path,
                            yaml_path=Config.yaml_data_path,
                            save_dir='../../../../dataset/chest',
                            num_classes=Config.num_classes,
                            class_names=Config.class_names,
                            is_train=False,
                            fold_path=fold_path,
                            duplicate_path=duplicate_path)

        exp_path = f'./detection/yolo/{yolo_ver}/runs/test/exp'
        if os.path.exists(exp_path): shutil.rmtree(exp_path)

        os.chdir(f'./detection/yolo/{yolo_ver}')

        infer_command = f'python \
        ./test.py \
        --batch-size {batch_size} \
        --img {image_size} \
        --conf {conf_thresh} \
        --iou {iou_thresh} \
        --weights {"../../../" + ck_path} \
        --data {"./../../../" + Config.yaml_data_path} \
        --augment \
        --save-txt \
        --save-conf \
        --device {device} \
        --exist-ok \
        --verbose'
       os.system(infer_command)

    elif mode == 'remote':
        df_valid = make_fold('test', Config.csv_path)
        test_image_dir = allocate_files(None, 
                            csv_path=Config.csv_path,
                            yaml_path=None,
                            save_dir='../../../dataset/chest',
                            num_classes=Config.num_classes,
                            class_names=Config.class_names,
                            is_train=False)
        
        exp_path = f'./detection/yolo/{yolo_ver}/runs/detect/exp'
        if os.path.exists(exp_path): shutil.rmtree(exp_path)

        os.chdir(f'./detection/yolo/{yolo_ver}')

        infer_command = f'python ./detect.py \
        --weights {"../../../" + ck_path} \
        --img {image_size} \
        --conf {conf_thresh} \
        --iou {iou_thresh} \
        --source {"../../../../" + test_image_dir} \
        --augment \
        --save-txt \
        --save-conf \
        --exist-ok \
        --nosave \
        --device {device}'
        os.system(infer_command)

    elif mode == 'pseudo':
	df_valid = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')
	test_image_dir = '../dataset/chest'
	allocate_pseudo_files(test_image_dir)

        exp_path = f'./detection/yolo/{yolo_ver}/runs/detect/exp'
        if os.path.exists(exp_path): shutil.rmtree(exp_path)

        os.chdir(f'./detection/yolo/{yolo_ver}')

	infer_command = f'python ./detect.py \
	--weights {ck_path} \
	--img {image_size} \
	--conf {conf_thresh} \
	--iou {iou_thresh} \
	--source {test_image_dir} \
	--augment \
	--save-txt \
	--save-conf \
	--exist-ok \
	--nosave'
	os.system(infer_commmand)

    os.chdir('../../..')
    prediction_path = os.path.join(exp_path, 'labels')
    df_sub = get_image_sub(prediction_path, df_valid)
    df_sub.to_csv(os.path.join(exp_path, 'image_sub.csv'), index=False)
    df_valid.to_csv(os.path.join(exp_path, 'valid.csv'), index=False)
    if mode == 'local':
        log.write('opacity map = %.5f\nnone map = %.5f\n'%map_2cls(df_valid, df_sub))

    log.write('Result saved to %s\n'%save_dir)
    t1 = time.time()
    log.write('Inference took %ds\n\n'%(t1 - t0))
    log.write('============================================================\n\n')

    return shutil.move(exp_path, save_dir)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck-paths', type=str, nargs='+')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--conf-thr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='remote')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Config.seed)
    
    opt = parse_opt()
    
    ck_paths, image_size, batch_size, \
    iou, conf, mode, device = \
    opt.ck_paths, opt.image_size, opt.batch_size, \
    opt.iou_thr, opt.conf_thr, opt.mode, opt.device

    for ck_path in ck_paths:
        yolo_infer(ck_path,
                   image_size=image_size,
                   batch_size=batch_size,
                   iou_thresh=iou,
                   conf_thresh=conf,
                   mode=mode,
                   save_dir='../result/yolo/submit',
                   fold_path=Config.fold_path,
                   duplicate_path=Config.duplicate_path,
                   device=device)

if __name__ == '__main__':
    main()
