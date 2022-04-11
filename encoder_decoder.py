from datetime import datetime

import torch
import torch.nn as nn

from decoder import DecoderRNN
from encoder import EncoderCNN
from utils import DEVICE, show_image, plot_attention

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, n_epochs, embed_size=300, attention_dim=256, decoder_dim=512, normalise=False, extractor="vgg", dropout=0.2):
        super().__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_epochs = n_epochs
        self.curr_epoch = 1
        self.normalise = normalise
        self.extractor = extractor
        self.fit_date = None

        self.dropout = dropout
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        self.encoder = EncoderCNN(extractor)

        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            dropout=dropout
        )


    def forward(self, images, captions):
        """
        Forward propagation.
        
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :param captions: captions, a tensor of dimensions (batch_size, vocab_size)
        :return: encoded images
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)

        return outputs

    def fit(self, data_loader, optimizer, loss_criterion, dataset):
        """
        Model training.
        
        :param dataset_loader: DataLoader for training
        :param optimizer: optimizer to update model's weights (Adam or other)
        :param loss_criterion: loss to use (cross entropy or other)
        :param dataset: object containing dataset information
        """
        loss = None
        self.fit_date = datetime.now().strftime("%Y_%m_%d_%H_%M")
        for epoch in range(self.curr_epoch, self.n_epochs + 1):
            for idx, (batch_images, batch_captions, _) in enumerate(iter(data_loader)):
                batch_images, batch_captions = batch_images.to(DEVICE), batch_captions.to(DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # Feed forward
                outputs, _ = self.forward(batch_images, batch_captions)

                # Loss
                targets = batch_captions[:, 1:]

                
                loss = loss_criterion(outputs.view(-1, self.vocab_size), targets.reshape(-1))

                # Backward pass
                loss.backward()

                # Update the optimizer
                optimizer.step()

                if idx % 100 == 0:
                    print(f"Epoch: {epoch} loss: {loss.item():.5f}")

                    if idx % 500 == 0:
                        # Generate the caption
                        img, _, img_name = next(iter(data_loader))
                        self.predict(img, dataset=dataset, img_name=img_name)

                    self.train()

            self.save(epoch)

        if loss is not None :
            return loss.item()

    def predict(self, features_tensors, dataset, img_name=None):
        """
        Get predicted caption and display image with MT score.
        
        :param features_tensors: batch of images
        :param dataset: object containing dataset information
        :param img_name: name of the first image of the batch into the dataset (optional)
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(features_tensors[0:1].to(DEVICE))
            captions, alphas = self.decoder.predict_caption(features, dataset.word2idx, dataset.idx2word)
            captions = captions[:-1]
            mt = ""
            if img_name is not None:

                captions_ref = dataset.df[dataset.df["image"] == img_name[0]]["caption"]
                captions_ref = [ caption.split() for caption in captions_ref]
                
                try:
                    print("nist nist nist")
                    print(captions_ref)
                    print(captions)
                    mt_score = sentence_nist(captions_ref, captions)
                except:
                    print("bleu bleu bleu")
                    print(captions_ref)
                    print(captions)
                    mt_score = sentence_bleu(captions_ref, captions)

                mt = f"\nMT score: {mt_score:.2f}"
            # print captions 
            print(captions)
            show_image(features_tensors[0], self.normalise, title=' '.join(captions)+mt)

        return captions, alphas

    def display_attention(self, data_loader, word2idx, idx2word, features_dims):
        """
        Display attention regions on images.
        
        :param data_loader: DataLoader from which images are taken
        :param word2idx: dictionary mapping words with unique indexes
        :param idx2word: dictionary mapping unique indexes with words
        :param features_dims: size of the input images
        """
        images, _, _ = next(iter(data_loader))

        img = images[0].detach().clone()
        captions, alphas = self.predict(img.unsqueeze(0), word2idx, idx2word)

        img = images[0].detach().clone()
        plot_attention(img, captions, alphas, self.normalise, features_dims)

    def save(self, num_epochs):
        """
        Save model checkpoint.
        
        :param num_epochs: number of epochs done until checkpoint
        """
        model_state = {
            "num_epochs": num_epochs,
            "embed_size": self.embed_size,
            "vocab_size": self.vocab_size,
            "attention_dim": self.attention_dim,
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "state_dict": self.state_dict()
        }

        save_name = f"{self.extractor}_{self.fit_date}.pth"
        torch.save(model_state, save_name)

    def load(self, saved_path):
        """
        Load model checkpoint.
        
        :param saved_path: path of the checkpoint to load
        """
        model_dict = torch.load(saved_path, map_location=torch.device(DEVICE))
        self.curr_epoch = model_dict["num_epochs"] + 1
        self.embed_size = model_dict["embed_size"]
        self.vocab_size = model_dict["vocab_size"]
        self.attention_dim = model_dict["attention_dim"]
        self.encoder_dim = model_dict["encoder_dim"]
        self.decoder_dim = model_dict["decoder_dim"]

        self.decoder = DecoderRNN(
            embed_size=self.embed_size,
            vocab_size=self.vocab_size,
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            dropout=self.dropout
        )

        self.load_state_dict(model_dict["state_dict"])
