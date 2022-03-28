import torch
import torch.nn as nn

from decoder import DecoderRNN
from encoder import EncoderCNN
from utils import show_image


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, normalise=False, extractor="vgg", dropout=0.2):
        super().__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.device = device
        self.normalise = normalise
        self.extractor = extractor

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
        features = self.encoder(images)
        outputs = self.decoder(features, captions)

        return outputs

    def fit(self, data_loader, dataset, optimizer, loss_criterion, epochs):
        for epoch in range(1, epochs + 1):
            for idx, (image, captions) in enumerate(iter(data_loader)):
                image, captions = image.to(self.device), captions.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Feed forward
                outputs, _ = self.forward(image, captions)

                # Loss
                targets = captions[:, 1:]
                loss = loss_criterion(outputs.view(-1, self.vocab_size), targets.reshape(-1))

                # Backward pass
                loss.backward()

                # Update the optimizer
                optimizer.step()

                if (idx + 1) % 100 == 0:
                    print(f"Epoch: {epoch} loss: {loss.item():.5f}")

                    # Generate the caption
                    img, _ = next(iter(data_loader))
                    self.predict(img, dataset)

                    self.train()

            self.save_model(epoch)

    def predict(self, features_tensors, dataset):
        self.eval()
        with torch.no_grad():
            features = self.encoder(features_tensors[0:1].to(self.device))
            caps, alphas = self.decoder.predict_caption(features, word2idx=dataset.word2idx, idx2word=dataset.idx2word)
            caption = ' '.join(caps)
            show_image(features_tensors[0], self.normalise, title=caption)

        return caps, alphas

    def save(self, num_epochs):
        model_state = {
            "num_epochs": num_epochs,
            "embed_size": self.embed_size,
            "vocab_size": self.vocab_size,
            "attention_dim": self.attention_dim,
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "state_dict": self.state_dict()
        }

        torch.save(model_state, "attention_model_state.pth")
