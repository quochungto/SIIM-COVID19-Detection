import sys
sys.path.append('.')

# stdlib
import os
import argparse

# custom
from utils.common import save_checkpoints
from utils.torch_common import seed_everything, memory_cleanup
from global_config import Config
from func import allocate_files

def yolo_train(folds, epochs, batch_size, image_size, weight='yolov5x', pseudo=False, device=0):
    result_path = '../result/yolo/checkpoints'
    output_paths = []
    pseudo_csv_path = '../dataset/pseudo.csv' if pseudo else None
    yolo_ver = 'yolov5' if 'yolov5' in weight else 'yolotrs' if 'yolotrs' in weight else None
    assert yolo_ver is not None

    for fold in folds:
        memory_cleanup()

        print(f'---- Training fold {fold} ----\n'.upper())
        print('Allocating files ... ')
        allocate_files(fold,
                        csv_path=Config.csv_path,
                        yaml_path=Config.yaml_data_path,
                        save_dir='../../../../dataset/chest',
                        num_classes=Config.num_classes,
                        class_names=Config.class_names,
                        is_train=True,
                        fold_path=Config.fold_path,
                        duplicate_path=Config.duplicate_path,
                        pseudo_csv_path=pseudo_csv_path)
        print('Done!\n')
        print('epochs=%d,batch_size=%d,image_size=%d,weight=%s,data_path=%s,hyp_path=%s,device=%s,pseudo=%s'\
              %(epochs,batch_size,image_size,weight,Config.yaml_data_path,Config.yaml_hyp_path,str(device),pseudo))

        os.chdir(f'./detection/yolo/{yolo_ver}')
        train_command = f'python3 ./train.py \
        --epochs {epochs} \
        --batch-size {batch_size} \
        --img {image_size} \
        --weights {weight}.pt \
        --data {"./../../../" + Config.yaml_data_path} \
        --hyp {"./../../../" + Config.yaml_hyp_path} \
        --device {device} \
        --cache'
        os.system(train_command)
        os.chdir('../../..')

        exp_path = f'./detection/yolo/{yolo_ver}/runs/train/exp'
        prefix = '%s_fold%d_bsize%d_%d_%d'%(yolo_ver, fold, batch_size, image_size, epochs)
        if pseudo_csv_path is not None: prefix += '_pseudo'
        output_paths.append(save_checkpoints(exp_path, result_path, prefix=prefix, mode='move'))
        print('result saved to %s\n'%output_paths[-1])

    return output_paths

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='yolov5x')
    parser.add_argument('--folds', type=str, nargs='+', default=[0, 1, 2, 3, 4], help='folds for training, i.e. 0 or 0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pseudo', action='store_true')
    parser.add_argument('--debug', action='store_true')
    #parser.add_argument('--infer', action='store_true')
    parser.add_argument('--redownload', action='store_true')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Config.seed)

    opt = parse_opt()
    
    weight, folds, epochs, \
    image_size, batch_size, \
    pseudo, debug, redownload, device = \
    opt.weight, opt.folds, opt.epochs, \
    opt.image_size, opt.batch_size, \
    opt.pseudo, opt.debug, opt.redownload, opt.device
    
    if debug:
        folds = [0]
        epochs = 1

    yolo_train(folds, epochs, batch_size, image_size, weight=weight, pseudo=pseudo, device=device)


if __name__ == '__main__':
    main()
