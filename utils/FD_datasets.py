import numpy as np
import sys
import os
from PIL import Image
import gzip
from torch.utils.data import Dataset

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
class EMNISTDataset(Dataset):
    def __init__(self, root, split='balanced', train=True, transform=None):
        self.root = root
        self.split = split
        self.train = train
        self.transform = transform

        if self.train:
            self.images_path = os.path.join(self.root, f'emnist-{self.split}-train-images-idx3-ubyte.gz')
            self.labels_path = os.path.join(self.root, f'emnist-{self.split}-train-labels-idx1-ubyte.gz')
        else:
            self.images_path = os.path.join(self.root, f'emnist-{self.split}-test-images-idx3-ubyte.gz')
            self.labels_path = os.path.join(self.root, f'emnist-{self.split}-test-labels-idx1-ubyte.gz')

        self.images, self.labels = self._load_data()
        self.len_dataset = len(self.labels)

    def _load_data(self):
        with gzip.open(self.images_path, 'rb') as img_path:
            images = np.frombuffer(img_path.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(self.labels_path, 'rb') as lbl_path:
            labels = np.frombuffer(lbl_path.read(), np.uint8, offset=8)
        return images, labels

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换成PIL Image对象以便应用transform
        image = Image.fromarray(image, mode='L')

        if self.transform is not None:
            image = self.transform(image)

        return image
class ClothingDataset_whole(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_files = []
        self.transform = transform
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('jpg', 'png', 'jpeg')):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('L')  # 转换为灰度图像
        if self.transform:
            image = self.transform(image)
        return image

