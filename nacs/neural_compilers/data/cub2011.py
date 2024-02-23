import os
import pandas as pd
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


class Cub2011(Dataset):
    #====================================================================
    # This implementation is heavily based on
    # https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
    #====================================================================
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
        
    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    
class Cub2011FS(Cub2011):
    def __init__(
        self, 
        root, 
        chosen_class,
        n_sample,
        seed=None,
        train=True, 
        transform=None, 
        loader=default_loader, 
        download=False
    ):
        super(Cub2011FS, self).__init__(root, train, transform, loader, download)
        if seed:
            np.random.seed(seed)
        targets = self.data.values[:, 2]
        chosen_idxs = np.array([], dtype=int)
        for cls in chosen_class:
            idxs = np.where(targets == cls + 1)[0]
            n_idxs = np.random.permutation(idxs)[0:n_sample]
            chosen_idxs = np.concatenate((chosen_idxs, n_idxs))
        permute_idxs = np.random.permutation(chosen_idxs)
        self.data = self.data.iloc[permute_idxs]
        # Map chosen classes to sequential numbers.
        chosen_class.sort()
        temp = self.data.values
        for idx, cls in enumerate(chosen_class, start=1):
            temp[:, 2] = np.where(temp[:, 2] == cls + 1, idx, temp[:, 2])
        self.data = pd.DataFrame(temp, columns=['img_id', 'filepath', 'target', 'is_training_img'])

# dataset = Cub2011FS('/cpfs/29cd2992fe666f2a/user/wangzekun/luojiawen/datasets', chosen_class=[10,25], n_sample=5)