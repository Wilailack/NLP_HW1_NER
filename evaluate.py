from conlleval import evaluate  # Reuse your existing conlleval functions
import torch

def unpad_and_convert(predictions, targets, ix2tag, pad_index=0):
    pred_all, true_all = [], []

    for pred_seq, true_seq in zip(predictions, targets):
        for pred_idx, true_idx in zip(pred_seq, true_seq):
            if true_idx == pad_index:
                break
            pred_all.append(ix2tag[pred_idx.item()])
            true_all.append(ix2tag[true_idx.item()])
    
    return true_all, pred_all

def eval_model(model, dataloader, ix2tag, device, pad_index=0):
    model.eval()
    all_preds = []
    all_tags = []

    with torch.no_grad():
        for word_batch, tag_batch in dataloader:
            word_batch = word_batch.to(device)
            tag_batch = tag_batch.to(device)

            output = model(word_batch)
            output = output.transpose(1, 2)  # (B, C, S)
            pred_ids = output.argmax(dim=1)  # (B, S)

            all_preds.extend(pred_ids.cpu())
            all_tags.extend(tag_batch.cpu())

    # Remove padding and convert indices to labels
    true_labels, pred_labels = unpad_and_convert(all_preds, all_tags, ix2tag, pad_index=pad_index)

    # Compute evaluation metrics
    return evaluate(true_labels, pred_labels, verbose=True)