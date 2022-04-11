from os.path import join

import pandas as pd
import spacy
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms

from utils import MAGIC_MU, MAGIC_SIGMA


class DatasetLoader(Dataset):
    """
    DataLoader.
    """

    def __init__(self, img_path, captions_file, normalise=False, img_size=299, nb_img=None):
        self.img_path = img_path
        if nb_img is not None:
            nb_img = int(nb_img) * 5

        self.df = pd.read_csv(captions_file)
        self.df = self.df[:nb_img]

        self.normalise = normalise

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.spacy_en = spacy.load("en_core_web_sm")
        self.word2idx = {"<START>": 0, "<END>": 1, "<UNK>": 2, "<PAD>": 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()  # scale image to [0,1]
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Get item.
        """

        caption = self.captions[idx]
        img_name = self.imgs[idx]

        # Image
        img = Image.open(join(self.img_path, img_name))
        img = self.transform(img)

        if self.normalise:
            for c in range(3):
                img[c] -= MAGIC_MU[c]
                img[c] /= MAGIC_SIGMA[c]

        # Captions
        caption = tensor([self.word2idx["<START>"]] +
                         self.tokenise(caption) +
                         [self.word2idx["<END>"]])

        return img, caption, img_name

    def build_vocab(self):
        """
        Build vocabulary.
        """

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
        """
        Tokenise sentence.

        :param sentence: sentence to tokenise
        """
        tokens = [str(token).lower() for token in self.spacy_en.tokenizer(sentence)]
        return [self.word2idx[tok] if tok in self.word2idx else self.word2idx["<UNK>"] for tok in tokens]
