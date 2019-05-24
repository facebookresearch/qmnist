## Pytorch dataset for QMNIST

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import print_function
import torch.utils.data as data
from torchvision.datasets.utils import download_url
from PIL import Image
import os
import os.path
import gzip
import lzma
import numpy as np
import torch
import codecs

class QMNIST(data.Dataset):
    """`QMNIST Dataset.
    Args:
        root (string): Root directory of dataset whose ``processed''
            subdir contains torch binary files with the datasets.
        what (string,optional): Can be 'train', 'test', 'test10k',
            'test50k', or 'nist' for respectively the mnist compatible
            training set, the 60k qmnist testing set, the 10k qmnist
            examples that match the mnist testing set, the 50k
            remaining qmnist testing examples, or all the nist
            digits. The default is to select 'train' or 'test'
            according to the compatibility argument 'train'.
        compat (bool,optional): A boolean that says whether the target
            for each example is class number (for compatibility with
            the MNIST dataloader) or a torch vector containing the
            full qmnist information. Default=True.
        download (bool, optional): If true, downloads the dataset from
            the internet and puts it in root directory. If dataset is
            already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform
            that takes in the target and transforms it.
        train (bool,optional,compatibility): When argument 'what' is
            not specified, this boolean decides whether to load the
            training set ot the testing set.  Default: True.

    """

    subsets = {
        'train':'train',
        'test':'test', 'test10k':'test', 'test50k':'test',
        'nist':'nist'
    }
    urls = {
        'train' : [ 'https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz',
                    'https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz' ] ,
        'test' :  [ 'https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz',
                    'https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz' ] ,
        'nist' :   [ 'https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-images-idx3-ubyte.xz',
                    'https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-labels-idx2-int.xz']
    }
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, what=None, compat=True,
                 transform=None, target_transform=None,
                 download=False, train=True):
        self.root = os.path.expanduser(root)
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        if what is None:
            what = 'train' if train else 'test'
        if not self.subsets.get(what):
            raise RuntimeError("Argument 'what' should be one of: \n  " +
                               repr(tuple(self.subsets.keys())) )
        self.what = what 
        self.compat = compat
        if not self._check_exists(what):
            self._process(what)
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, what + '.pt'))

    def __getitem__(self, index):
        """
        Args: 
            index (int): Index Returns a tuple (image, target). 
        When compat is true, the target is the class number.
        Otherwise the target is a torch vector with all the qmnist
        information, that is, the class number, the nist hsf
        partition, the writer id, the digit id for this writer, the
        class ascii code, the global digit id, the duplicate id, and a
        reserved field.  The duplicate id is always zero in the
        'train' and 'test' splits. It may be nonzero in the 'nist'
        split and indicates that this digit is a duplicate of another
        one.  There are only three duplicate digits in the nist
        dataset.
        """
        img = Image.fromarray(self.data[index].numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        if self.compat:
            target = target[0].item()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img,target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self, what):
        return os.path.exists(os.path.join(self.processed_folder, what + ".pt"))
           
    def _process(self, what):
        if not self._check_exists(what):
            os.makedirs(self.processed_folder, exist_ok=True)
            mydir = os.path.dirname(os.path.realpath(__file__))
            file = self.subsets[what]
            urls = self.urls[file]
            assert urls
            myfiles = []
            # get files (either in current dir, or download them)
            for url in urls:
                filename = url.rpartition('/')[2]
                file_path = os.path.join(mydir, filename)
                if not os.path.isfile(file_path):
                    # File not found in the same dir as qmnist.py
                    file_path = os.path.join(self.raw_folder, filename)
                    if not os.path.isfile(file_path):
                        if self.download:
                            os.makedirs(self.raw_folder, exist_ok=True)
                            print('Downloading ', url, ' into ', file_path)
                            download_url(url, root=self.raw_folder, filename=filename, md5=None)
                        else:
                            raise RuntimeError("Dataset '" + file + "' not found." +
                                               '  Use download=True to download it')
                myfiles.append(file_path)
            # process and save as torch files
            data = read_idx3_ubyte(myfiles[0])
            targets = read_idx2_int(myfiles[1])
            if what == 'test10k':
                data = data[0:10000,:,:]
                targets = targets[0:10000,:]
            if what == 'test50k':
                data = data[10000:,:,:]
                targets = targets[10000:,:]
            with open(os.path.join(self.processed_folder, what + '.pt'), 'wb') as f:
                torch.save((data, targets), f)

    def __repr__(self):
        fmt_str =  'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.what)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def open_maybe_compressed_file(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    elif path.endswith('.xz'):
        return lzma.open(path, 'rb')
    else:
        return open(path,'rb')
    
def read_idx2_int(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 12*256 + 2
        length = get_int(data[4:8])
        width = get_int(data[8:12])
        parsed = np.frombuffer(data, dtype=np.dtype('>i4'), offset=12)
        return torch.from_numpy(parsed.astype('i4')).view(length,width).long()

def read_idx3_ubyte(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 8 * 256 + 3
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

