from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset import FaceLandmarksDataset
from transforms import Rescale, RandomCrop, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

# def show_landmarks(image, landmarks):
#     """Show image with landmarks"""
#     plt.imshow(image)
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#     plt.pause(0.001)  # pause a bit so that plots are updated

# face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
#                                     root_dir='data/faces/')

# fig = plt.figure()

# scale = Rescale(256)
# crop = RandomCrop(128)
# composed = transforms.Compose([Rescale(256),
#                                RandomCrop(224)])

# # Apply each of the above transforms on sample.
# fig = plt.figure()
# sample = face_dataset[65]
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)

#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)

# plt.show()

# for i in range(len(face_dataset)):
#   sample = face_dataset[i]

#   print(i, sample['image'].shape, sample['landmarks'].shape)

#   ax = plt.subplot(1, 4, i + 1)
#   plt.tight_layout()
#   ax.set_title('Sample #{}'.format(i))
#   ax.axis('off')
#   show_landmarks(**sample)

#   if i == 3:
#     plt.show()
#     break

wait = input("PRESS ENTER TO CONTINUE.")