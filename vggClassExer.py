import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2+0.5 # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# Steps to training an Image classifier
# 1. Load and normalize the datasets with torchvision
# 2. Define a CNN
# 3. Define a loss function
# 4. Train the network on training datasets
# 5. Test the network on the test datasets


#Load an Normalize below
if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainSet = torchvision.datasets.CIFAR10(root='./data',
                                            train = True,
                                            download = True,
                                            transform = transform
                                            )
    trainloader = torch.utils.data.DataLoader(trainSet,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers=2
                                              )
    testSet = torchvision.datasets.CIFAR10(root='./data',
                                           train = False,
                                           download = True,
                                           transform = transform
                                           )
    testLoader = torch.utils.data.DataLoader(testSet,
                                             batch_size=batch_size,
                                             shuffle = False,
                                             num_workers=2
                                             )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #gets some random training images
    dataIter = iter(trainloader)
    images, labels = dataIter.next()

    #print the image labels below
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    #display images
    imshow(torchvision.utils.make_grid(images))
    #print the image labels below
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))