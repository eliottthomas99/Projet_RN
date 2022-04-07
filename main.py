import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import DatasetLoader
from encoder_decoder import EncoderDecoder
from utils import DEVICE, collate

PATH = "flickr8k/"
NORMALISE = True
MODEL_PARAMS = {
    "encoder_channels": {
        "vgg16": 512,
        "resnet50": 2048,
        "inception_v3": 2048
    },
    "features_dims": {
        "vgg16": 7,
        "resnet50": 8,
        "inception_v3": 8
    }
}

# Hyperparameters
extractor = "inception_v3"
batch_size = 32
embed_size = 300
attention_dim = 256
encoder_dim = MODEL_PARAMS["encoder_channels"][extractor]
decoder_dim = 512
learning_rate = 1e-3
epochs = 10


def main():
    # Load data
    dataset = DatasetLoader(img_path=PATH + "Images/",
                            captions_file=PATH + "captions.txt",
                            normalise=NORMALISE)
    dataset.build_vocab()

    vocab_size = len(dataset.word2idx)
    pad_idx = dataset.word2idx["<PAD>"]

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, pad_idx)
    )

    print("device:", DEVICE)

    # Initialize model
    model = EncoderDecoder(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        normalise=NORMALISE,
        extractor=extractor,
        n_epochs=epochs
    ).to(DEVICE)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    # model.load("vgg_2022_04_06_14_01.pth")
    model.fit(data_loader, optimizer, ce_loss, dataset)

    # Display attentions
    #model.display_attention(data_loader, dataset.word2idx, dataset.idx2word, features_dims=MODEL_PARAMS["features_dims"][extractor])


if __name__ == "__main__":
    main()
