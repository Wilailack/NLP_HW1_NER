import torch.nn as nn

class BiLSTMTagger(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers, bidirectional, num_classes):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.lstm = nn.LSTM(
            input_size=embeddings.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, word):

        # S = Sentence Length
        # B = Batch Size
        # E = Embedding Dimension 300
        # C = Classes Number
        # H = Hidden Size
        
        # word (B,S) (32,39)
        embedded = self.embedding(word) 
        # embedd (B,S) (32,39,300)
        out, (h, c) = self.lstm(embedded) 
        # out (B,S,H) (32,39,256)
        out = self.fc(out)
        out[:,:,0] = float("-inf") # ignore pad index in prediction
        # out (B,S,C) (32,39,22)

        return out
