import torch
import numpy as np

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def list_data(data):
    sentences = [s.strip().split("\n") for s in data.strip().split("\n\n")]
    all_words, all_tags, sentence_words, sentence_tags = [], [], [], []

    for sentence in sentences:
        words, tags = [], []
        for line in sentence:
            if not line.strip():
                continue
            word, tag = line.split("\t")
            all_words.append(word)
            all_tags.append(tag)
            words.append(word)
            tags.append(tag)
        sentence_words.append(words)
        sentence_tags.append(tags)

    return all_words, all_tags, sentence_words, sentence_tags

def lowercase(sentences):
    return [[word.lower() for word in sentence] for sentence in sentences]

def build_vocab(words, pad_token="<pad>", unk_token="<unk>"):
    vocab = set(words)
    vocab.update([pad_token, unk_token])
    ix2word = sorted(vocab)
    word2ix = {word: ix for ix, word in enumerate(ix2word)}
    return word2ix, ix2word

def build_tag_map(tags, pad_token="<pad>"):
    tag_set = set(tags)
    tag_set.add(pad_token)
    tag2ix = {tag: ix for ix, tag in enumerate(sorted(tag_set))}
    ix2tag = {ix: tag for tag, ix in tag2ix.items()}
    return tag2ix, ix2tag

def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    return [seq + ["<pad>"] * (max_len - len(seq)) for seq in sequences]

def convert_word_to_ids(batch_sentences, word2idx):
    return [[word2idx.get(word, word2idx["<unk>"]) for word in sentence] for sentence in batch_sentences]

def convert_tag_to_ids(batch_tags, tag2idx):
    return [[tag2idx.get(tag, tag2idx["<pad>"]) for tag in tags]for tags in batch_tags]

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, word_ids, tag_ids):
        self.word_ids = torch.tensor(word_ids, dtype=torch.long)
        self.tag_ids = torch.tensor(tag_ids, dtype=torch.long)

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        return self.word_ids[idx], self.tag_ids[idx]

def load_glove(path, word_to_index, embedding_dim=300):
    embeddings = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector_values = values[1:]
            # Skip lines with wrong number of dimensions
            if len(vector_values) != embedding_dim:
                continue
            try:
                vector = np.asarray(vector_values, dtype='float32')
                embeddings[word] = vector
            except ValueError:
                # This catches things like: could not convert string to float: '.'
                continue
    # Default vectors for <pad> and <unk>
    pad_vector = np.zeros(embedding_dim, dtype='float32')
    unk_vector = np.random.uniform(-0.25, 0.25, embedding_dim).astype('float32')
    embedding_matrix = np.zeros((len(word_to_index), embedding_dim), dtype='float32')
    for word, idx in word_to_index.items():
        if word == "<pad>":
            embedding_matrix[idx] = pad_vector
        elif word == "<unk>":
            embedding_matrix[idx] = unk_vector
        else:
            embedding_matrix[idx] = embeddings.get(word, unk_vector)
    return torch.tensor(embedding_matrix, dtype=torch.float32)