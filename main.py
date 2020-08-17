#Import required packages
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import math
import sys
plt.ion()   # interactive mode

# IMPORT OWN CLASSES
# testtesttest

from DS_NN_models import Image_Caption_Dataset, Rescale, RandomCrop, ToTensor

#Setup the main code
if __name__ == '__main__':
    #Define the transforms for the images
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])

    # Define the dataset
    insta_dataset = Image_Caption_Dataset(csv_path = 'instagram_data\captions_csv.csv',
                                          image_path = 'instagram_data',
                                          transform=composed)



    #It is important that the length of the captions for each batch have the same length

    #Get the different lengths of each caption
    indices = insta_dataset.get_train_indices()
    #Sample from indices with a certain length
    initial_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)

    #Define the dataloader and use the sampler defined abouve
    dataloader = DataLoader(insta_dataset, num_workers=4,
                            batch_sampler=torch.utils.data.sampler.BatchSampler(sampler=initial_sampler,
                                                                    batch_size=insta_dataset.batch_size,
                                                                    drop_last=False))

    # Import the NNs
    from DS_NN_models import Encoder,Decoder


    #Some hyperparameters
    embed_size = 256
    hidden_size = 100
    num_layers =1
    num_epochs = 1
    vocab_size = len(insta_dataset.vocab)


    #Fix the datatype for the layers
    encoder = Encoder(embed_size)
    encoder = encoder.double()
    decoder = Decoder(embed_size,hidden_size,vocab_size,num_layers)
    decoder = decoder.double()

    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # TODO #3: Specify the learnable parameters of the model.
    params = list(decoder.parameters()) + list(encoder.embed.parameters())

    # TODO #4: Define the optimizer.
    optimizer = torch.optim.Adam(params = params, lr = 0.001)

    # Set the total number of training steps per epoch.
    num_batches = math.ceil(insta_dataset.__len__() / dataloader.batch_sampler.batch_size)
    num_batches = 500
    save_every = 1

    #Run the main loop

    #Loop over the epochs
    for epoch in range(num_epochs):
        #Loop over each batch in that epoch
        for batch in range(num_batches):
            # Load the data by iterating the dataloader
            images, captions = next(iter(dataloader))
            # Fix the datatype
            captions = captions.long()
            # Remove the stored gradients
            decoder.zero_grad()
            decoder.zero_grad()

            # Run the encoder
            features = encoder(images.double())

            # Run the decoder
            outputs = decoder(features, captions)

            # (You can call the encoder() / decoder () but what is really run is the encoder.forward() / decoder.forward() )


            # Calculate the loss
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Print and stuff
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch+1, num_epochs, batch+1, num_batches, loss.item(), np.exp(loss.item()))

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), 'decoder-%d.pkl' % epoch)
            torch.save(encoder.state_dict(), 'encoder-%d.pkl' % epoch)
