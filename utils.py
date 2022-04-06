import matplotlib.pyplot as plt
from torch import cat, cuda
from torch import device as torch_device
from torch.nn.utils.rnn import pad_sequence

MAGIC_MU = [0.485, 0.456, 0.406]
MAGIC_SIGMA = [0.229, 0.224, 0.225]

DEVICE = torch_device('cuda:0' if cuda.is_available() else "cpu")


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


def plot_attention(img, caption, alphas, normalise=False):
    # Unnormalise
    if normalise:
        for i in range(3):
            img[i] *= MAGIC_SIGMA[i]
            img[i] += MAGIC_MU[i]
    
    img = img.numpy().transpose((1, 2, 0))
    img_cpy = img

    fig = plt.figure(figsize=(15, 15))

    len_caption = len(caption)
    for l in range(len_caption):
        att = alphas[l].reshape(7, 7)
        
        ax = fig.add_subplot(len_caption // 2, len_caption // 2, l+1)
        ax.set_title(caption[l])
        img = ax.imshow(img_cpy)
        ax.imshow(att, cmap='gray', alpha=0.7, extent=img.get_extent())
        

    plt.tight_layout()
    plt.show()
