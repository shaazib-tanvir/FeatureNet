from models.bagnet import bagnet33
import torch.nn as nn
import numpy as np
import torch
import math
import gensim.downloader


class AverageFeaturesGRUDecoder(nn.Module):
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

        return output


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

        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout, batch_first=True)
        self.embedding = nn.Embedding(len(vocabulary), hidden_size, padding_idx=0, _freeze=True)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        self.output_fc = nn.Linear(hidden_size, len(vocabulary))
        self.softmax = nn.Softmax(dim=2)

        """
        if train:
            print("Loading pretrained embedding...")
            glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")
            with torch.no_grad():
                for (index, word) in vocabulary.idx2word.items():
                    if word in glove_vectors:
                        embedding_tensor = torch.FloatTensor(np.copy(glove_vectors.get_vector(word)))
                        self.embedding.weight[index] = embedding_tensor
                        self.embedding._freeze = False
        """

    def generate_mask(self, sequence_length):
        return nn.Transformer.generate_square_subsequent_mask(sequence_length, device="cuda")

    def forward(self, features, words):
        sequence = self.positional_encoding(self.embedding(words))
        mask = self.generate_mask(sequence.size()[1])
        report = self.transformer(features, sequence, tgt_mask=mask, tgt_is_causal=True)
        report = self.output_fc(report)
        report = self.softmax(report)
        report = report.permute(0, 2, 1)

        return report


class FeatureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.bagnet = bagnet33(pretrained=True, num_classes=self.num_classes)

    def forward(self, image):
        prediction, average_features, features = self.bagnet(image)
        features = features.view(-1, features.size()[1], features.size()[2] ** 2)
        features = features.permute(0, 2, 1)

        return prediction, average_features, features
