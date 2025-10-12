import torch

def train(model, dataloader, optimizer, criterion, device):
    
    model.train()
    epoch_loss = 0
    num = 0

    for word, tag in dataloader:
        word, tag = word.to(device), tag.to(device)
        optimizer.zero_grad() # clear the gradients

        # forward
        predicted_tag = model(word) 
        predicted_tag = torch.transpose(predicted_tag, 1, 2)  # (B, C, S)

        # compute the batch loss
        loss = criterion(predicted_tag, tag)

        # choose the maximum index
        predicted_tag_ = predicted_tag.argmax(dim=1)

        # backward (calculate the gradients)
        loss.backward()

        # gradient descent or Adam step (update the parameters)
        optimizer.step()

        epoch_loss += loss.item()
        num += 1
        epoch_loss_ = epoch_loss / num

    return epoch_loss, epoch_loss_, predicted_tag_, tag
