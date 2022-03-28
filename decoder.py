import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.W1 = nn.Linear(encoder_dim, attention_dim)
        self.W2 = nn.Linear(decoder_dim, attention_dim)

        self.V = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
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
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, dropout=0.2):
        super().__init__()

        self.device = device
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
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions):
        # Vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch, decoder_dim)

        # Get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.dropout(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def predict_caption(self, features, max_len=20, word2idx=None, idx2word=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch, decoder_dim)

        word = torch.tensor(word2idx['<START>']).view(1, -1).to(self.device)
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
