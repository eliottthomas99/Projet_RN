import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


class Attention(nn.Module):
    """
    Attention layer.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature dimension of encoded images
        :param decoder_dim: dimension of the decoder's RNN
        :param attention_dim: size of the attention network
        """
        super().__init__()

        self.W1 = nn.Linear(encoder_dim, attention_dim) # (batch, layers, attention_dim)
        self.W2 = nn.Linear(decoder_dim, attention_dim) # (batch, attention_dim)

        self.V = nn.Linear(attention_dim, 1) # (batch, layers, 1)

    def forward(self, features, hidden_state):
        """
        :param features: encoded images, a tensor of size (batch_size, num_pixels, encoder_dim)
        :param hidden_state: previous hidden state of the decoder, a tensor of size (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        scores = torch.tanh(
            # (batch, layers, attention_dim) + (batch, attention_dim)
            self.W1(features) + self.W2(hidden_state).unsqueeze(1)
        )   # (batch, layers, attemtion_dim)

        scores = self.V(scores)     # (batch, layers, 1)
        scores = scores.squeeze(2)  # (batch, layers)

        alpha = F.softmax(scores, dim=1)  # (batch, layers)

        context = features * alpha.unsqueeze(2)  # (batch, layers, features_dim)
        context = context.sum(dim=1)             # (batch, layers)

        return alpha, context


class DecoderRNN(nn.Module):
    """
    Decoder 
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.2):
        """
        :param embed_size: dimension of word embeddings
        :param vocab_size: size of the vocabulary
        :param attention_dim: size of attention network
        :param encoder_dim: feature dimension of encoded images
        :param decoder_dim: dimension of the decoder's RNN
        :param dropout: dropout
        """
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of size (batch_size, num_pixels, encoder_dim)

        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions):
        """
        Forward propagation.

        :param features: encoded images, a tensor of size (batch_size, num_pixels, encoder_dim)
        :param captions: encoded captions, a tensor of size (batch_size, max_len)
        :return: prediction and attention probabilities
        """

        # Vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch, decoder_dim)

        # Get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(DEVICE)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(DEVICE)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.dropout(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def predict_caption(self, features, word2idx=None, idx2word=None, max_len=20):
        """
        Predict the caption for a given image.

        :param features: encoded images, a tensor of size (batch_size, num_pixels, encoder_dim)
        :param word2idx: word to index mapping
        :param idx2word: index to word mapping
        :param max_len: maximum length of the caption
        :return: predicted caption
        """
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch, decoder_dim)

        word = torch.tensor(word2idx['<START>']).view(1, -1).to(DEVICE)
        embeds = self.embedding(word)

        alphas, captions = list(), list()

        for _ in range(max_len):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.dropout(h))
            output = output.view(batch_size, -1)

            alphas.append(alpha.cpu().detach().numpy())

            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())

            # End of sequence
            if idx2word[predicted_word_idx.item()] == "<END>":
                break

            # Send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # Return corresponding words
        return [idx2word[idx] for idx in captions], alphas
