import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = 20  # max caption length

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (batch_size, 1, embed_size)

        for _ in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # (batch_size, vocab_size)
            _, predicted = outputs.max(1)                # (batch_size)
            sampled_ids.append(predicted.item())

            if predicted.item() == 2:  # Assuming 2 is <end> token index
                break

            inputs = self.embed(predicted)               # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                 # (batch_size, 1, embed_size)

        return sampled_ids

