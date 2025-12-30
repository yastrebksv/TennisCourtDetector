import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def train(model, train_loader, optimizer, criterion, device, epoch, max_iters=1000):
    model.train()
    losses = []
    max_iters = min(max_iters, len(train_loader))

    # Create tqdm progress bar for training
    pbar = tqdm(enumerate(train_loader), total=max_iters, desc=f'Epoch {epoch} [Train]')
    for iter_id, batch in pbar:
        out = model(batch[0].float().to(device))
        gt_hm_hp = batch[1].float().to(device)
        loss = criterion(F.sigmoid(out), gt_hm_hp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        losses.append(loss.item())
        if iter_id > max_iters:
            break

    return np.mean(losses)





