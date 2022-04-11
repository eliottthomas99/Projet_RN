import click
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import DatasetLoader
from encoder_decoder import EncoderDecoder
from optisearch import optisearch
from utils import DEVICE, MODEL_PARAMS, NORMALISE, PATH, collate, MAGIC_MU, MAGIC_SIGMA
from PIL import Image


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
@click.option("-ld", "--load", default=None, help="path to the model to load")
@click.option("-ip", "--img_path", default=None, help="path to the image to predict")
def main(extractor, batch_size, embed_size, attention_dim, decoder_dim, learning_rate, dropout, nb_img, epochs, tuna, load, img_path):
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
    """
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
    #loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loss = nn.NLLLoss(ignore_index=pad_idx)


    if tuna:
        optisearch(extractor, dataset, data_loader, loss, vocab_size, encoder_dim, epochs, NORMALISE)
    else:
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if load is not None:
            model.load(load)

        if img_path is not None:
            print("Predicting:", img_path, "before training")
            # Image preprocessing
            img = Image.open(img_path)
            img = dataset.transform(img)

            for c in range(3):
                img[c] -= MAGIC_MU[c]
                img[c] /= MAGIC_SIGMA[c]

            img = img.unsqueeze(0)
            _ = model.predict(img, dataset)


        # Train model
        model.fit(data_loader, optimizer, loss, dataset)

        if img_path is not None:
            print("Predicting:", img_path, "after training")
            _ = model.predict(img, dataset)
 

    # Display attentions
    # model.display_attention(data_loader, dataset.word2idx, dataset.idx2word, features_dims=MODEL_PARAMS["features_dims"][extractor])


if __name__ == "__main__":
    main()
