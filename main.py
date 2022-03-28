import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch import device as torch_device
from torch.utils.data import DataLoader

from data_loader import DatasetLoader
from encoder_decoder import EncoderDecoder
from utils import collate

PATH = "flickr8k/"
NORMALISE = True
ENCODER_DIMS = {
    "vgg": 512,
    "resnet": 2048
}

# Hyperparameters
extractor = "vgg"
batch_size = 20
embed_size = 300
attention_dim = 256
encoder_dim = ENCODER_DIMS[extractor]
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

    device = torch_device('cuda:0' if cuda.is_available() else "cpu")
    print("device:", device)

    # Initialize model
    model = EncoderDecoder(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        device=device,
        normalise=NORMALISE,
        extractor=extractor
    ).to(device)

    # Loss and optimizer
    loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    model.fit(data_loader, dataset, optimizer, loss_criterion, epochs)


if __name__ == "__main__":
    main()
