import matplotlib.pyplot as plt
from torch import cat
from torch.nn.utils.rnn import pad_sequence


def collate(batch, pad_idx):
    images = [item[0].unsqueeze(0) for item in batch]
    images = cat(images, dim=0)

    captions = [item[1] for item in batch]
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, captions


def show_image(img, title=None):
    # Unnormalise
    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406

    print(img.shape)
    img = img.numpy().transpose((1, 2, 0))

    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.pause(1e-3)
