"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image
import os, glob
import random
import torch
from torchvision import transforms

class FaceDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.add_argument('--same_prob', type=float, default=0.8, help='')
        # parser.set_defaults(max_dataset_size=10000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;

        self.root = opt.dataroot
        self.same_prob = opt.same_prob
        self.count_subj = 0
        self.image_paths = []
        self.image_ids = []
        print("="*90)
        for idx, data_path in enumerate(opt.dataset_name):
            if opt.dataset_type[idx]=='face_file': ## IF the dataset dir contain only images, each image represent different identity
                data_folder = os.path.join(self.root, data_path + "/images256x256")
                for filename in glob.glob(data_folder + "/*"):
                    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                        self.image_paths.append(filename)
                        self.image_ids.append(self.count_subj)
                        self.count_subj +=1
            elif opt.dataset_type[idx]=='face_folder': ## If the dataset dir contain multiple folders of images, each folder represent different identity
                data_folder = os.path.join(self.root, data_path + "/images256x256")
                for folder_name in glob.glob(data_folder + "/*"):
                    for filename in glob.glob(folder_name + "/*"):
                        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                            self.image_paths.append(filename)
                            self.image_ids.append(self.count_subj)
                    self.count_subj +=1
        # import pdb; pdb.set_trace()
        # print(len(self.image_paths))
        # print(self.image_paths[:5])
        # print(count_subj)
        # self.transform = get_transform(opt)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ])
        print(self.transform)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        ###
        s_idx = index 
        if random.random() > self.same_prob:
            f_idx = index
        else:
            f_idx = random.randrange(len(self.image_paths))

        if self.image_ids[f_idx] == self.image_ids[s_idx]:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.image_paths[f_idx])
        s_img = Image.open(self.image_paths[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)

        return {'src_data': f_img, 'tgt_data': s_img, 'same': same, 'src_paths':self.image_paths[f_idx], 'tgt_paths': self.image_paths[s_idx]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
