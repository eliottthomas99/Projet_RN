import click
from matplotlib import pyplot as plt
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader

from data_loader import DatasetLoader
from encoder_decoder import EncoderDecoder
from optisearch import optisearch
from utils import DEVICE, MODEL_PARAMS, NORMALISE, collate, plot_history


# Command line arguments
@click.command()
@click.argument("extractor", type=click.Choice(["vgg16", "resnet50", "inception_v3"]))
@click.option("-dt", "--data_path", default="data/flickr8k/", help="default is flickr8k/")
@click.option("-bs", "--batch_size", default=32, help="default is 32")
@click.option("-es", "--embed_size", default=300, help="default is 300")
@click.option("-ad", "--attention_dim", default=256, help="default is 256")
@click.option("-dd", "--decoder_dim", default=512, help="default is 512")
@click.option("-lr", "--learning_rate", default=1e-3, help="default is 1e-3")
@click.option("-dp", "--dropout", default=0.2, help="default is 0.2")
@click.option("-nb", "--nb_img", default=None, help="default is None --> all images")
@click.option("-e", "--epochs", default=10, help="default is 10")
@click.option("-t", "--tuna", default=0, help="default is 0")
@click.option("-ld", "--load", default=None, help="path to the model to load")
@click.option("-ip", "--img_path", default=None, help="path to the image to predict")
@click.option("-t", "--test_path", default=None, help="path to the test set")
@click.option("-att", "--disp_attention", default=0, help="default is 0")
@click.option("-hist", "--plot_loss", default=0, help="default is 0")
def main(extractor, data_path, batch_size, embed_size, attention_dim, decoder_dim, learning_rate, dropout, nb_img, epochs, tuna, load, img_path, test_path, disp_attention, plot_loss):
    """
    Main function.

    :param extractor: extractor to use
    :param batch_size: batch size
    :param embed_size: embedding size
    :param attention_dim: attention dimension
    :param decoder_dim: decoder dimension
    :param learning_rate: learning rate
    :param dropout: dropout
    :param nb_img: number of images to use
    :param epochs: number of epochs
    :param tuna: if 1, use the tuner to fine tune hyperparameters
    :param load: path to the model to load (optional)
    :param img_path: path to the image to predict
    :param disp_attention: if 1, display attention weights
    """
    encoder_dim = MODEL_PARAMS[extractor]["encoder_channels"]

    # Load data
    dataset = DatasetLoader(
        img_path=data_path + "/Images/",
        captions_file=data_path + "/captions.txt",
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

    if tuna:
        optisearch(extractor, dataset, data_loader, loss, vocab_size, encoder_dim, epochs, NORMALISE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if load is not None:
            model.load(load)
            model.to(DEVICE)

        # Predict caption
        if img_path is not None:
            # Image preprocessing
            img = Image.open(img_path)
            img = dataset.transform(img)
            img = img.unsqueeze(0)

            print("Predicting:", img_path, "before training")
            _ = model.predict(img, dataset)

        # Train model
        model.fit(data_loader, optimizer, loss, dataset)
        
        # Test model
        if test_path is not None:
            # Load data
            dataset_test = DatasetLoader(
                img_path=test_path + "/Images/",
                captions_file=test_path + "/captions.txt",
                normalise=NORMALISE,
                nb_img=nb_img
            )
            dataset_test.build_vocab()

            pad_idx_test = dataset_test.word2idx["<PAD>"]

            data_loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=batch_size,
                num_workers=2,
                shuffle=True,
                collate_fn=lambda batch: collate(batch, pad_idx_test)
            )
            print(model.test(data_loader_test, loss))
        
        # Plot loss history
        if plot_loss:
            plot_history(model.loss_history)

        # Predict caption
        if img_path is not None:
            print("Predicting:", img_path, "after training")
            _ = model.predict(img, dataset)

        # Display attentions
        if disp_attention:
            model.display_attention(data_loader, dataset)


if __name__ == "__main__":
    main()  # for command line usage
