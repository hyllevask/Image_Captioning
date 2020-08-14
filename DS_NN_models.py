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
from vocab import Vocabulary
from skimage import io, transform
import nltk
#import tqdm
plt.ion()   # interactive mode

class Image_Caption_Dataset(Dataset):
    def __init__(self, csv_path, image_path, transform=None, batch_size = 4):
        self.captionsfile = pd.read_csv(csv_path)
        self.image_path = image_path
        self.transform = transform
        self.vocab = Vocabulary(vocab_threshold=2)
        self.batch_size = batch_size
        all_tokens = [nltk.tokenize.word_tokenize(str(self.captionsfile.iloc[index,2]).lower()) for index in range(len(self.captionsfile))]
        self.caption_lengths = [len(token) for token in all_tokens]

    def __len__(self):
        return len(self.captionsfile)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.image_path + '\\' + self.captionsfile.iloc[idx, 1] + '.jpg'
        image = io.imread(image_name)
        caption = self.captionsfile.iloc[idx, 2]



        try:
            words = nltk.tokenize.word_tokenize(caption.lower())
        except:
            caption = str(caption)
            words = nltk.tokenize.word_tokenize(caption.lower())

        id_caption = [0]
        for word in words:
            try:
                id_caption.append(self.vocab.word2idx[word])
            except:
                id_caption.append(2)
        id_caption.append(1)

        sample = {'image': image, 'caption': np.array(id_caption)}
        if self.transform:
            sample = self.transform(sample)

        return sample['image'],sample['caption']
    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, caption = sample['image'], sample['caption']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'caption': caption}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, caption = sample['image'], sample['caption']
        ##
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'caption': caption}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, caption = sample['image'], sample['caption']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'caption': caption}

class Encoder(nn.Module):
    # Encodes the images into feature vectors
    def __init__(self, embedded_size):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embedded_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features



class Decoder(nn.Module):
    def __init__(self, embedded_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedded_size)
        self.lstm = nn.LSTM(
            input_size=self.embedded_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).double(),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).double())

    def forward(self, features, captions):
        captions = captions[:, :-1]
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embedding(captions)
        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.fc(lstm_out)
        return outputs

    def Predict(self, inputs, max_len=20):
        final_output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_idx = torch.max(outputs, dim=1)
            final_output.append(max_idx.cpu().numpy()[0].item())
            if (max_idx == 1 or len(final_ouput) >= 20):
                break

            inputs = self.word_embedding(max_idx)
            inputs = inputs.unsqueeze(1)
        return final_output