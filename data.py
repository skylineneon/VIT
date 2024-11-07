from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F


class MnistDataset(Dataset):

    def __init__(self, data_dir):

        self.datas = []

        for tag_dir in os.listdir(data_dir):
            img_path = f"{data_dir}/{tag_dir}"
            for img_name in os.listdir(img_path):
                img = Image.open(f"{img_path}/{img_name}")
                img = torch.tensor(np.array(img), dtype=torch.float32)

                self.datas.append((img, torch.tensor(int(tag_dir))))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        _img, _tag = self.datas[index]

        _x = _img[None]
        # _t =F.one_hot(_tag,10)

        _x = _x / 255.
        # _t = _t.to(torch.float32)

        return _x, _tag


if __name__ == '__main__':
    dataset = MnistDataset("datas/train")
    x, t = dataset[0]
    print(x, t)
