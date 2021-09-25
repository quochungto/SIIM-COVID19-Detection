import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-paths', type=str, nargs='+', help='paths to predicted csv', required=True)# img-level paths
    parser.add_argument('-ws', type=float, nargs='+', help='image-level ensemble weights', required=True)# img-level weights
    parser.add_argument('--iou-thr', type=float, default=0.6, help='boxes fusion iou threshold')# iou thres
    parser.add_argument('--conf-thr', type=float, default=0.0001, help='boxes fusion skip box threshold')# conf thes
    parser.add_argument('--none-thr', type=float, default=0.6, help='theshold for hard-labeling images as none-class')
    parser.add_argument('--opacity-thr', type=float, default=0.095, help='threshold for hard-labeling images as opacity-class')

    return parser.parse_args()

def main():
    t0 = time.time()
    
    opt = parse_opt()

    assert len(opt.paths) == len(opt.ws), f'len(paths) == {len(ws)}'
    
    # logging
    log = Logger()
    os.makedirs('../logging', exist_ok=True)
    log.open(os.path.join('../logging', 'post_processing.txt'), mode='a')
    log.write('STUDY-LEVEL\n')
    log.write('weight\tpath\n')
    for p, w in zip(opt.study_csv, opt.sw):
        log.write('%.2f\t%s\n'%(w, p))
    log.write('\n')
    log.write('IMAGE-LEVEL\n')
    log.write('weight\tpath\n')
    for p, w in zip(opt.image_csv, opt.iw):
        log.write('%.2f\t%s\n'%(w, p))
    log.write('\n')
    log.write('iou_thr=%.4f,skip_box_thr=%.4f\n'%(opt.iou_thr, opt.conf_thr))

    # prepare data    
    dfs = [pd.read_csv(df_path) for df_path in opt.paths]
    with open(opt.std2img, 'rb') as f:
        std2img = pickle.load(f)

    with open(opt.img2shape, 'rb') as f:
        img2shape = pickle.load(f)
    ids, hs, ws = [], [], []
    for k, v in img2shape.items():
        ids.append(k + '_image')
        hs.append(v[0])
        ws.append(v[1])
    df_meta = pd.DataFrame({'id': ids, 'w': ws, 'h': hs})

    # post-process
    df_study = ensemble_study(dfs_study, weights=opt.sw)
    df_image = ensemble_image(dfs_image, df_meta, mode='wbf', \
            iou_thr=opt.iou_thr, skip_box_thr=opt.conf_thr, weights=opt.iw)[0]
    df_image, df_none = postprocess_image(df_image, df_study, std2img) 
    df_study = postprocess_study(df_study, df_none, std2img)

    df_sub = pd.concat([df_study, df_image], axis=0, ignore_index=True)
    df_sub = df_sub[['id', 'PredictionString']]
    df_sub.to_csv('../result/submission/submission.csv', index=False)

    # logging
    log.write('Number of boxes per image on average: %d\n'%check_num_boxes_per_image(df=df_sub))
    t1 = time.time()
    log.write('Post-process took %ds\n\n'%(t1 - t0))
    log.write('Submission saved to ./result/submission/submission.csv\n\n')
    log.write('============================================================\n\n')

if __name__ == '__main__':

    main()
