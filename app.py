import nltk
nltk.download('punkt')
from utils.data_loader import get_loader
from utils.caption_preprocessing import Vocabulary

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
import torch
import torchvision.transforms as transforms
import config

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader, vocab = get_loader(transform=transform)

    encoder = EncoderCNN(config.embed_size).to(config.device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(config.device)

    # Optionally load training loop or evaluation
    print("Model Initialized.")

if __name__ == "__main__":
    main()



