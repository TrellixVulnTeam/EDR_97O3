import os
import os.path
import hashlib
import errno
import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F
from termcolor import cprint
import numpy as np
from scipy.special import softmax

def download_url(url, root, filename = None, md5 = None):

    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                urllib.request.urlretrieve(url, fpath)

def check_integrity(fpath, md5 = None):

    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def lrt_correction(targets, outputs, current_thd = 0.3, thd_increment = 0.1):

    corrected_count = 0
    y_noise = torch.tensor(targets).clone()
    output = outputs.max(1)[0]
    output_arg = outputs.argmax(1)
    LR = []
    for i in range(len(y_noise)):
        LR.append(float(outputs[i][int(y_noise[i])]/output[i]))

    for i in range(int(len(y_noise))):
        if LR[i] < current_thd:
            y_noise[i] = output_arg[i]
            corrected_count += 1

    if corrected_count < 0.001*len(y_noise) and current_thd != 0.9:
        current_thd += thd_increment
        current_thd = min(current_thd, 0.9)
        cprint("Update thresold Value -> {}\n".format(current_thd), "cyan")

    return y_noise, current_thd
