import os
import shutil
import zipfile
import argparse
import kaggle
import pandas as pd

def download_kaggle_datasets(datasets, force=False, quiet=False):
    for dset in datasets:
        dset_name = os.path.split(dset)[-1]
        save_path = os.path.join('../dataset', dset_name)
        if not os.path.exists(save_path) or force:
            kaggle.api.dataset_download_files(dset, path=save_path, force=force, quiet=quiet, unzip=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true', help='force redownload')
    parser.add_argument('-q', action='store_true', help='suppress verbose output')
    opt = parser.parse_args()
    download_kaggle_datasets([
                            'quochungto/fold-split-siim',
                            'quochungto/1024x1024-png-siim',
                            'quochungto/metadatasets-siim',
                            'quochungto/image-level-psuedo-label-metadata-siim',
                            # pseudo data
                            'quochungto/files-for-psuedo-label-siim',
                            'raddar/ricord-covid19-xray-positive-tests',
                            'quochungto/covid19-posi-dump-siim',
                            # yolotrs
                            'quochungto/yolotr-pretrained',
                            # mmdet
                            'vgarshin/mmdet-vfnet-pretrained'
                            ], force=opt.f, quiet=opt.q)

    # train dataste
    csv_path = '../dataset/metadatasets-siim/meta_1024_colab.csv'
    temp = pd.read_csv(csv_path)
    temp['filepath'] = temp['filepath'].apply(lambda x: x.replace('/content/kaggle_datasets', '../dataset'))
    #if debug: temp = temp[:15]
    temp.to_csv('../dataset/meta.csv', index=False)

    # ricord dataset
    df = pd.read_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')
    df['filepath'] = df['filepath'].apply(lambda x: x.replace('/kaggle/input', '../dataset'))
    df.to_csv('../dataset/image-level-psuedo-label-metadata-siim/bimcv_ricord.csv')
 
    # pseudo datasets
    pseudo_csv_path = '../dataset/files-for-psuedo-label-siim/image_level_pseudo_low_none_06000_high_none_06000_op_00950.csv'
    temp = pd.read_csv(pseudo_csv_path)
    temp['filepath'] = temp['colab_filepath'].apply(lambda x: x.replace('/content/kaggle_datasets', '../dataset'))
    #if debug: temp = temp[:15]
    temp.to_csv('../dataset/pseudo.csv', index=False)

    #if not opt.q:
    #    print('All datasets:', *os.listdir("../dataset"), sep='\n')

if __name__ == '__main__':
    main()
