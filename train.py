# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.data_loader import get_loader
import config

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_loader, vocab = get_loader(transform=transform)

    encoder = EncoderCNN(config.embed_size).to(config.device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(config.device)
    print(" Models loaded.")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=config.learning_rate)

    total_steps = len(data_loader)

    for epoch in range(config.num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for i, (images, captions) in enumerate(loop):
            images = torch.stack(images).to(config.device)
            captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).to(config.device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            features = encoder(images)
            outputs = decoder(features, inputs)

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / total_steps
        print(f" Epoch [{epoch+1}/{config.num_epochs}] completed. Avg Loss: {avg_loss:.4f}")

        torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
        torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pth")
        torch.save(vocab.word2idx, f"vocab_epoch_{epoch+1}.pth")
        print(f" Saved model for epoch {epoch+1}.")

if __name__ == "__main__":
    train()
