import matplotlib.pyplot as plt
from torch import cat
from torch.nn.utils.rnn import pad_sequence

MAGIC_MU = [0.485, 0.456, 0.406]
MAGIC_SIGMA = [0.229, 0.224, 0.225]


def collate(batch, pad_idx):
    images = [item[0].unsqueeze(0) for item in batch]
    images = cat(images, dim=0)

    captions = [item[1] for item in batch]
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, captions


def show_image(img, normalise, title=None):
    # Unnormalise
    if normalise:
        for i in range(3):
            img[i] *= MAGIC_SIGMA[i]
            img[i] += MAGIC_MU[i]

    img = img.numpy().transpose((1, 2, 0))

    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.pause(1e-3)
