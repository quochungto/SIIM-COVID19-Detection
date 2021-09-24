import os
import shutil
import time

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def save_checkpoints(src, dst, prefix=None, mode='copy'):
    t = time.strftime('%Y%m%d_%H%M%S')
    if prefix is not None:
        t = prefix + '_' + t
    save_path = os.path.join(dst, t)
    if mode == 'copy':
        shutil.copytree(src, save_path)
    elif mode == 'move':
        shutil.move(src, save_path)

    return save_path


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines = f.readlines()
    strings=[]
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()

        if s != '':
            strings.append(s)

    return strings


def get_size(start_path = '.', unit='b'):
    assert os.path.exists(start_path), 'path does not exist'
    unit_divide = {'b': 1, 'kb': 2**10, 'mb': 2**20, 'gb': 2**30, 'tb': 2**40}
    total_size = 0
    if os.path.isfile(start_path):
        total_size = os.stat(start_path).st_size
    else:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

    return total_size / unit_divide[unit]
