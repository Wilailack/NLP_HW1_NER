import torch

def predict(model, dataloader, ix2tag, sentences, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            pred = output.argmax(dim=2).cpu()

            for tokens, pred_ids in zip(sentences, pred):
                tags = [ix2tag[i.item()] for i in pred_ids[:len(tokens)]]
                results.append(list(zip(tokens, tags)))

    return results

def save_predictions(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in results:
            for word, tag in sentence:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")
