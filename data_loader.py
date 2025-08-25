import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from utils.caption_preprocessing import Vocabulary
from nltk.tokenize import TreebankWordTokenizer


class FlickrDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.vocab = Vocabulary(threshold=5)
        self.captions = []
        self.image_ids = []
        tokenizer = TreebankWordTokenizer()

        # Read the captions.txt file
        with open("dataset/captions.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Handle various separators
                if ',' in line:
                    parts = line.split(',', 1)
                elif '\t' in line:
                    parts = line.split('\t')
                elif ' ' in line:
                    parts = line.split(' ', 1)
                else:
                    continue

                if len(parts) != 2:
                    continue

                img_id, caption = parts
                img_id = img_id.split('#')[0].strip()
                caption = caption.strip()

                # Validate image file existence
                image_path = os.path.join("dataset", "Images", img_id)
                if not os.path.isfile(image_path):
                    print(f" Skipping missing image: {img_id}")
                    continue

                tokens = tokenizer.tokenize(caption.lower())
                self.captions.append(tokens)
                self.image_ids.append(img_id)

        print(f" Loaded {len(self.captions)} captions and {len(self.image_ids)} images.")

    def __getitem__(self, idx):
        image_path = os.path.join("dataset", "Images", self.image_ids[idx])
        image = Image.open(image_path).convert('RGB')

        caption = self.captions[idx]
        if self.transform is not None:
            image = self.transform(image)

        vocab = self.vocab
        caption = [vocab.word2idx.get(word, vocab.word2idx['<unk>']) for word in caption]
        caption = [vocab.word2idx['<start>']] + caption + [vocab.word2idx['<end>']]

        return image, torch.tensor(caption)

    def __len__(self):
        return len(self.captions)


def get_loader(transform=None):
    dataset = FlickrDataset(transform=transform)

    if len(dataset) == 0:
        print(" ERROR: No image-caption pairs found. Check paths or caption format.")
        exit()

    vocab = dataset.vocab
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: list(zip(*x))
    )

    return data_loader, vocab
     