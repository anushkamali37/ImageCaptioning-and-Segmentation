# generate.py
import torch
from torchvision import transforms
from PIL import Image
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.caption_preprocessing import Vocabulary
import config
import os

# Load transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_vocab(vocab_path): 
    word2idx = torch.load(vocab_path)
    vocab = Vocabulary(threshold=1)
    vocab.word2idx = word2idx  
    vocab.idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab

def load_model(encoder_path, decoder_path, vocab):
    encoder = EncoderCNN(config.embed_size).to(config.device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab.word2idx), config.num_layers).to(config.device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=config.device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=config.device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def generate_caption(image_path, encoder, decoder, vocab):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(config.device)

    with torch.no_grad():
        features = encoder(image)
        output = decoder.sample(features)

    sentence = []
    for idx in output:
        word = vocab.idx2word.get(idx, "<unk>")
        if word == "<end>":
            break
        if word != "<start>":
            sentence.append(word)

    return ' '.join(sentence)

if __name__ == "__main__":
    # Update this with your test image
    image_path = "dataset/Images/47871819_db55ac4699.jpg"  
    encoder_path = "encoder_epoch_1.pth"
    decoder_path = "decoder_epoch_1.pth"
    vocab_path = "vocab_epoch_1.pth"

    # Load vocab and models
    vocab = load_vocab(vocab_path)
    encoder, decoder = load_model(encoder_path, decoder_path, vocab)

    # Generate caption
    caption = generate_caption(image_path, encoder, decoder, vocab)
    print(f"\n Image: {os.path.basename(image_path)}")
    print(f" Caption: {caption}")
