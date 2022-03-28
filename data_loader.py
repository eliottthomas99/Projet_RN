from os.path import join

import pandas as pd
import spacy
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetLoader(Dataset):

    def __init__(self, img_path, captions_file, normalise=False):
        self.img_path = img_path
        self.df = pd.read_csv(captions_file)

        self.normalise = normalise
        self.magic_mu = [0.485, 0.456, 0.406]
        self.magic_sigma = [0.229, 0.224, 0.225]

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.spacy_en = spacy.load("en_core_web_sm")
        self.word2idx = {"<START>": 0, "<END>": 1, "<UNK>": 2, "<PAD>": 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor()  # scale image to [0,1]
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]

        # Image
        img = Image.open(join(self.img_path, img_name))
        img = self.transform(img)

        if self.normalise:
            for c in range(3):
                img[c] -= self.magic_mu[c]
                img[c] /= self.magic_sigma[c]

        # Captions
        caption = tensor([self.word2idx["<START>"]] + self.tokenise(caption) + [self.word2idx["<END>"]])

        return img, caption

    def build_vocab(self):
        curr_idx = len(self.word2idx)
        freq_threshold = 2
        frequencies = dict()

        for sentence in self.captions:
            tokens = [str(token).lower() for token in self.spacy_en.tokenizer(sentence)]

            for tok in tokens:
                if tok in frequencies.keys():
                    frequencies[tok] += 1
                else:
                    frequencies[tok] = 1

                if frequencies[tok] == freq_threshold:
                    self.word2idx[tok] = curr_idx
                    self.idx2word[curr_idx] = tok
                    curr_idx += 1

    def tokenise(self, sentence):
        tokens = [str(token).lower() for token in self.spacy_en.tokenizer(sentence)]
        return [self.word2idx[tok] if tok in self.word2idx else self.word2idx["<UNK>"] for tok in tokens]
