import torch
from data_utils import *
from model import BiLSTMTagger
from train import train
from evaluate import eval_model
from predict import predict, save_predictions
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--------------------------------> Using device: {device}")

    print("--------------------------------> Loading data and preprocessing...")
    # Load data
    data_train = read_data("data/train.txt")
    data_val = read_data("data/dev.txt")
    data_test = read_data("data/test-submit.txt")

    # Preprocess data
    train_words, train_tags, train_sentences_words, train_sentences_tags = list_data(data_train)
    _, _, val_sentences_words, val_sentences_tags = list_data(data_val)
    test_sentences_words = [s.strip().split("\n") for s in data_test.strip().split("\n\n")]

    # Lowercase
    train_sentences_words = lowercase(train_sentences_words)
    val_sentences_words = lowercase(val_sentences_words)
    test_sentences_words = lowercase(test_sentences_words)

    # print(f"len(train_words): {len(train_words)}")
    # print(f"len(train_tags): {len(train_tags)}")
    # print(f"len(train_sentences_words): {len(train_sentences_words)}")
    # print(f"len(train_sentences_tags): {len(train_sentences_tags)}")
    # print(f"len(val_sentences_words): {len(val_sentences_words)}")
    # print(f"len(val_sentences_tags): {len(val_sentences_tags)}")
    # print(f"len(test_sentences_words): {len(test_sentences_words)}")
    # print(f"Example train sentence words: {train_sentences_words[0]}")
    # print(f"Example train sentence tags: {train_sentences_tags[0]}")
    # print(f"tag unique: {set(train_tags)}")

    # Lowercase all words for vocab building
    train_words = [w.lower() for w in train_words]

    # Build vocab and tag maps
    word2idx, idx2word = build_vocab(train_words)
    tag2idx, idx2tag = build_tag_map(train_tags)

    # print(f"<pad> ID : {word2idx['<pad>']}, <unk> ID: {word2idx['<unk>']}")
    # print(f"tag of index 0 : {idx2tag[0]}")
    # print(f"tag2idx: {tag2idx}")

    # Convert to IDs and pad sequences
    batch_train_words = pad_sequences(train_sentences_words)
    batch_train_tags = pad_sequences(train_sentences_tags)
    batch_val_words = pad_sequences(val_sentences_words)
    batch_val_tags = pad_sequences(val_sentences_tags)
    batch_test_words = pad_sequences(test_sentences_words)

    # print(f"batch_train_words[0]: {batch_train_words[0]}")
    # print(f"batch_train_tags[0]: {batch_train_tags[0]}")
    # print(f"len(batch_train_words[0]): {len(batch_train_words[0])}")

    # Convert words and tags to IDs
    X_train = convert_word_to_ids(batch_train_words, word2idx)
    y_train = convert_tag_to_ids(batch_train_tags, tag2idx)
    X_val = convert_word_to_ids(batch_val_words, word2idx)
    y_val = convert_tag_to_ids(batch_val_tags, tag2idx)
    X_test = convert_word_to_ids(batch_test_words, word2idx)

    # print(f"X_train[0]: {X_train[0]}")
    # print(f"y_train[0]: {y_train[0]}")


    # Create datasets and dataloaders
    train_dataset = NERDataset(X_train, y_train)
    val_dataset = NERDataset(X_val, y_val)
    test_dataset = torch.tensor(X_test) # Convert test data to tensor - can not use NERDataset since no tags

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    print(f"--------------------------------> Number of training samples: {len(train_dataset)}")
    print(f"--------------------------------> Number of validation samples: {len(val_dataset)}")
    print(f"--------------------------------> Number of test samples: {len(test_dataset)}")

    # Load GloVe
    print("--------------------------------> Loading GloVe embeddings...")
    embeddings = load_glove("glove.840B.300d.txt", word2idx)
    print(f"--------------------------------> Embeddings shape: {embeddings.shape}")

    # Model
    model = BiLSTMTagger(embeddings, hidden_size=256, num_layers=2,
                         bidirectional=True, num_classes=len(tag2idx)).to(device)

    # Training
    print("--------------------------------> Training model...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tag2idx["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_plot = []
    loss_plot = []

    print(f"---------------------------------------------------------")

    for epoch in range(100):
        _, epoch_loss_, _, _ = train(model, train_loader, optimizer, criterion, device)

        print(f"Epoch: {epoch+1} | Train Loss iter: {epoch_loss_:.4f}")

        epoch_plot.append(epoch+1)
        loss_plot.append(epoch_loss_)

    # Plot training loss
    plt.plot(epoch_plot, loss_plot)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig("train-loss.png")    
    print("--------------------------------> Training complete.")

    # Evaluation
    print("--------------------------------> Evaluating on validation set...")
    eval_model(model, val_loader, idx2tag, device, pad_index=tag2idx["<pad>"])

    # Save model
    # torch.save(model.state_dict(), "ner-model.pt")

    # Test predictions
    print("--------------------------------> Predicting on test set...")
    # model.load_state_dict(torch.load("ner-model.pt"))
    results = predict(model, test_loader, idx2tag, test_sentences_words, device)
    save_predictions(results, "data/test-predictions.txt")
    print("--------------------------------> Test predictions saved to test-predictions.txt")

if __name__ == "__main__":
    main()
