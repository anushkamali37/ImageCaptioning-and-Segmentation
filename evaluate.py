import torch
import torchvision.transforms as transforms
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.caption_preprocessing import Vocabulary
import config
import os

# Load Vocabulary
vocab = Vocabulary(threshold=5)
vocab.word2idx = torch.load("vocab_epoch_1.pth")
vocab.idx2word = {idx: word for word, idx in vocab.word2idx.items()}

# Load Models
encoder = EncoderCNN(config.embed_size).to(config.device)
decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab.word2idx), config.num_layers).to(config.device)
encoder.load_state_dict(torch.load("encoder_epoch_1.pth", map_location=config.device))
decoder.load_state_dict(torch.load("decoder_epoch_1.pth", map_location=config.device))
encoder.eval()
decoder.eval()

# Preprocess Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(config.device)

def generate_caption(image_tensor):
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features)
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id.item()]
            if word == '<end>':
                break
            if word != '<start>':
                sampled_caption.append(word)
    return sampled_caption

def evaluate_bleu_score(image_path, reference_caption):
    image_tensor = load_image(image_path)
    predicted_caption = generate_caption(image_tensor)

    # Tokenize reference and prediction
    reference_tokens = reference_caption.lower().strip().split()
    candidate_tokens = predicted_caption

    # Compute BLEU Score
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

    print("Reference:", ' '.join(reference_tokens))
    print("Predicted:", ' '.join(candidate_tokens))
    print(f"\n BLEU Score: {bleu_score:.4f}")


image_path = "dataset/Images/47871819_db55ac4699.jpg"
reference_caption = "on ground two women players are playing football"
evaluate_bleu_score(image_path, reference_caption)
