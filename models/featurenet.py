from models.bagnet import bagnet33
import torch.nn as nn
import numpy as np
import torch
import math
import gensim.downloader


"""class AverageFeaturesGRUDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocabulary_size, dropout=0.1):
        super().__init__()
        self.decoder = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.fc = nn.Linear(hidden_size, vocabulary_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, average_features, words):
        tokens = self.dropout(self.embedding(words))
        output, _ = self.decoder(tokens, average_features)
        output = self.fc(output).permute(0, 2, 1)
        output = self.softmax(output)

        return output"""


class PositionalEncoding(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 dropout: float):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding):
        maxlen = token_embedding.size()[1]
        den = torch.exp(-torch.arange(0, self.embedding_size, 2) * math.log(10000) / self.embedding_size).to("cuda")
        pos = torch.arange(0, maxlen, device="cuda").reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, self.embedding_size), device="cuda")
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        return self.dropout(token_embedding + pos_embedding[:token_embedding.size(0), :])


class FeatureTransformer(nn.Module):
    def __init__(self, hidden_size, nhead, num_layers, dropout, vocabulary, train=True):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.embedding = nn.Embedding(len(vocabulary), hidden_size, padding_idx=0, _freeze=True)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        self.output_fc = nn.Linear(hidden_size, len(vocabulary))
        self.softmax = nn.Softmax(dim=2)

    def generate_mask(self, sequence, sequence_length):
        mask = sequence == 0
        padding_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill_(mask, float("-inf"))
        return nn.Transformer.generate_square_subsequent_mask(sequence_length, device="cuda"), padding_mask

    def forward(self, features, words):
        mask, padding_mask = self.generate_mask(words, words.size()[1])
        sequence = self.positional_encoding(self.embedding(words))
        report = self.transformer_decoder(sequence, features, tgt_mask=mask, tgt_key_padding_mask=padding_mask, tgt_is_causal=True)
        report = self.output_fc(report)
        report = self.softmax(report)
        report = report.permute(0, 2, 1)

        return report


class FeatureNet(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()

        self.num_classes = num_classes
        self.bagnet = bagnet33(pretrained=True, num_classes=self.num_classes)

    def forward(self, image):
        prediction, features = self.bagnet(image)
        features = features.reshape(-1, 1, features.shape[1])

        return prediction, features
