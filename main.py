import click
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import DatasetLoader
from encoder_decoder import EncoderDecoder
from utils import DEVICE, collate

# Constants
PATH = "flickr8k/"
NORMALISE = True
MODEL_PARAMS = {
    "vgg16": {
        "encoder_channels": 512,
        "features_dims": 7
    },
    "resnet50": {
        "encoder_channels": 2048,
        "features_dims": 8
    },
    "inception_v3": {
        "encoder_channels": 2048,
        "features_dims": 8
    }
}


# Command line arguments
@click.command()
@click.argument("extractor", type=click.Choice(["vgg16", "resnet50", "inception_v3"]))
@click.option("-bs", "--batch_size", default=32, help="default is 32")
@click.option("-es", "--embed_size", default=300, help="default is 300")
@click.option("-ad", "--attention_dim", default=256, help="default is 256")
@click.option("-dd", "--decoder_dim", default=512, help="default is 512")
@click.option("-lr", "--learning_rate", default=1e-3, help="default is 1e-3")
@click.option("-e", "--epochs", default=10, help="default is 10")
def main(extractor, batch_size, embed_size, attention_dim, decoder_dim, learning_rate, epochs):
    encoder_dim = MODEL_PARAMS[extractor]["encoder_channels"]

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
    # model.display_attention(data_loader, dataset.word2idx, dataset.idx2word, features_dims=MODEL_PARAMS["features_dims"][extractor])


if __name__ == "__main__":
    main()
