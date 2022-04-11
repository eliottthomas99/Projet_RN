import click
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import DatasetLoader
from encoder_decoder import EncoderDecoder
from optisearch import optisearch
from utils import DEVICE, collate, MODEL_PARAMS, PATH, NORMALISE


# Command line arguments
@click.command()
@click.argument("extractor", type=click.Choice(["vgg16", "resnet50", "inception_v3"]))
@click.option("-bs", "--batch_size", default=32, help="default is 32")
@click.option("-es", "--embed_size", default=300, help="default is 300")
@click.option("-ad", "--attention_dim", default=256, help="default is 256")
@click.option("-dd", "--decoder_dim", default=512, help="default is 512")
@click.option("-lr", "--learning_rate", default=1e-3, help="default is 1e-3")
@click.option("-dp", "--dropout", default=0.2, help="default is 0.2")
@click.option("-nb", "--nb_img", default=None, help="default is None --> all images")
@click.option("-e", "--epochs", default=10, help="default is 10")
@click.option("-t", "--tuna", default=0, help="default is 0")
def main(extractor, batch_size, embed_size, attention_dim, decoder_dim, learning_rate, dropout, nb_img, epochs, tuna):
    encoder_dim = MODEL_PARAMS[extractor]["encoder_channels"]

    # Load data
    dataset = DatasetLoader(
        img_path=PATH + "Images/",
        captions_file=PATH + "captions.txt",
        normalise=NORMALISE,
        nb_img=nb_img
    )
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
        n_epochs=epochs,
        dropout=dropout
    ).to(DEVICE)

    # Loss and optimizer
    loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # loss = nn.L1Loss()


    if tuna:
        optisearch(extractor, dataset, data_loader, loss, vocab_size, encoder_dim, epochs, NORMALISE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Train model
        # model.load("vgg_2022_04_06_14_01.pth")
        model.fit(data_loader, optimizer, loss, dataset)

    # Display attentions
    # model.display_attention(data_loader, dataset.word2idx, dataset.idx2word, features_dims=MODEL_PARAMS["features_dims"][extractor])


if __name__ == "__main__":
    main()
