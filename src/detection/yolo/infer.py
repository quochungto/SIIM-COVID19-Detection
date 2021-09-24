import argparse
from global_config import Config
from func import yolo_infer

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ck-paths', type=str, nargs='+')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--conf-thr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--redownload', action='store_true')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

def main():
    seed_everything(Config.seed)
    
    opt = parse_opt()
    
    # parser
    ck_paths, image_size, batch_size, \
    iou, conf, mode, debug, \
    redownload, device = \
    opt.ck_paths, opt.image_size, opt.batch_size, \
    opt.iou_thr, opt.conf_thr, opt.mode, opt.debug, \
    opt.redownload, opt.device

    ## download datasets
    #datasets = [
    #        'quochungto/4-june-2021', 
    #        'quochungto/1024x1024-png-siim',
    #        'quochungto/metadatasets-siim',
    #        'quochungto/yolov5-official-v50']
    #print('============== downloading datasets ==============')
    #download_kaggle_datasets(datasets, force=redownload)
    #print(f'downloaded datasets: {os.listdir(INPUT_BASE)}')

    ## correct file paths
    #csv_path = os.path.join(INPUT_BASE, 'metadatasets-siim/meta_1024_colab.csv')
    #temp = pd.read_csv(csv_path)
    #temp['filepath'] = temp['filepath'].apply(lambda x: x.replace('/content/kaggle_datasets', INPUT_BASE))
    #if debug: 
    #    temp = pd.concat([temp[temp['split']=='train'][:15], temp[temp['split']=='test'][:5]])
    #temp.to_csv(os.path.join(INPUT_BASE, 'meta.csv'), index=False)

    ## install yolov5
    #if not os.path.exists(os.path.join(WORKING_BASE, 'yolov5')):
    #    print('Installing Yolov5 ... ', end='', flush=True)
    #    copy_and_overwrite(os.path.join(INPUT_BASE, 'yolov5-official-v50/yolov5-5.0'), os.path.join(WORKING_BASE, 'yolov5'))
    #    os.chdir(os.path.join(WORKING_BASE, 'yolov5'))
    #    os.system('pip install -q -r requirements.txt')
    #    os.chdir(WORKING_BASE)
    #    print('Done!')

    # infer
    #if debug:
    #    folds = [0]
    #    epochs = 1

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
