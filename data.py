import kagglehub

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import h5py
import pandas as pd

import os
from glob import glob
from config import *
import random

from concurrent.futures import ThreadPoolExecutor


def loadSeveral(filePaths, loadFunc, message):
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
        for i, result in enumerate(executor.map(loadFunc, filePaths)):
            print(f"\rLoaded {i + 1}/{len(filePaths)} {message} files", end="")
            yield result


class BraTSData(Dataset):
    def __init__(self, config: Config):
        self.config = config
        if config.train:
            # if "cache" in config:
            #     os.environ['KAGGLEHUB_CACHE_DIR'] = config.cache
            self.path = kagglehub.dataset_download("awsaf49/brats2020-training-data", output_dir=config.cache)
        else:
            self.path = config.path

        # Looks like every file is valid, not sure why everyone had a check
        sliceFiles = glob(os.path.join(self.path, "**", "*.h5"), recursive=True)
        # print("Checking valid slices...")
        # indices = []
        # validPaths = []
        # sliceData = loadSeveral(sliceFiles, self.loadSlice, "slice")
        # for i, data in enumerate(sliceData):
        #     if data is not None:
        #         indices.append(i)
        #         validPaths.append(sliceFiles[i])

        # print(len(indices), len(sliceFiles))

        self.indices = np.arange(len(sliceFiles))
        self.validPaths = sliceFiles

        metadata = pd.read_csv(glob(os.path.join(self.path, "*Metadata.csv"), recursive=True)[0])
        self.metadata = metadata
        self.volumeNames = self.metadata.volume.unique()

    def loadSlice(self, path):
        with h5py.File(path, 'r') as f:
            image = f['image'][()] 
            mask = f['mask'][()] 
    
        # Convert from one hot encoding to categorical labels
        if mask.ndim == 3:
            target = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
            target[mask[:,:,0] == 1] = 1 
            target[mask[:,:,1] == 1] = 2
            target[mask[:,:,2] == 1] = 3
        else:
            # Map from (0, 1, 2, 4) to (0, 1, 2, 3)
            target = np.copy(mask).astype(np.int64)
            target[mask == 4] = 3
        
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(target).long()

        return image, mask

    def loadVolume(self, name):
        images = []
        masks = []
        indices = []
        for i, file in enumerate(glob(os.path.join(self.path, "**", f"*volume_{name}*.h5"), recursive=True)):
            data = self.loadSlice(file)
            if data is not None:
                image, mask = data
                images.append(image)
                masks.append(mask)
                # Get slice index from file name, which is in the format "volume_{name}_slice_{index}.h5"
                indices.append(os.path.basename(file).split("_")[-1].removesuffix(".h5"))

        # Sort by slice index
        compiled = zip(indices, zip(images, masks))
        compiled = sorted(compiled, key=lambda x: int(x[0]))
        images, masks = zip(*[item[1] for item in compiled])

        return torch.stack(images), torch.stack(masks)
    
    def volumeCrop(self, image, mask, patchSize=128, bias=0.67):
        ph, pw, pd = patchSize, patchSize, patchSize
        h, w, d = mask.shape

        if random.random() < bias:
            # Pick a random foreground voxel as the center
            foregroundVoxels = torch.nonzero(mask > 0)  # (N, 3)
            if len(foregroundVoxels) > 0:
                center = foregroundVoxels[random.randint(0, len(foregroundVoxels) - 1)]
                x = center[0].item() - ph // 2
                y = center[1].item() - pw // 2
                z = center[2].item() - pd // 2
            else:
                bias = 0  # fall through to random
        
        else:  # pure background sample
            x = random.randint(0, h - ph)
            y = random.randint(0, w - pw)
            z = random.randint(0, d - pd)

        # Clamp so the patch stays inside the volume
        x = max(0, min(x, h - ph))
        y = max(0, min(y, w - pw))
        z = max(0, min(z, d - pd))

        return image[:, x:x+ph, y:y+pw, z:z+pd], mask[x:x+ph, y:y+pw, z:z+pd]

    def __len__(self):
        if self.config.trainingSet == "volumetric":
            return len(self.volumeNames) * self.config.patchesPerVolume

        elif self.config.trainingSet == "slices":
            return len(self.indices)
    
    def __getitem__(self, key):
        if self.config.trainingSet == "volumetric":
            volumeIdx = key % len(self.volumeNames)
            image, mask = self.loadVolume(self.volumeNames[volumeIdx])
            return self.volumeCrop(image, mask)

        elif self.config.trainingSet == "slices":
            image, mask = self.loadSlice(self.validPaths[key])
            return image, mask


if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "config.json"))
    dataset = BraTSData(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, masks in dataloader:
        print(images.shape)
        print(masks.shape)
        break
