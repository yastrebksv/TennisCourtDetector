import torch
import numpy as np
import torch.nn.functional as F
from utils import is_point_in_image
from scipy.spatial import distance
from postprocess import postprocess
from dataset import courtDataset
from tracknet import BallTrackerNet
import argparse
import torch.nn as nn

def val(model, val_loader, criterion, device, epoch):
    model.eval()
    losses = []
    tp, fp, fn, tn = 0, 0, 0, 0
    max_dist = 7
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            batch_size = batch[0].shape[0]
            out = model(batch[0].float().to(device))
            kps = batch[2]
            gt_hm = batch[1].float().to(device)
            loss = criterion(F.sigmoid(out), gt_hm)

            pred = F.sigmoid(out).detach().cpu().numpy()
            for bs in range(batch_size):
                for kps_num in range(14):
                    heatmap = (pred[bs][kps_num] * 255).astype(np.uint8)
                    x_pred, y_pred = postprocess(heatmap)
                    x_gt = kps[bs][kps_num][0].item()
                    y_gt = kps[bs][kps_num][1].item()

                    if is_point_in_image(x_pred, y_pred) and is_point_in_image(x_gt, y_gt):
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < max_dist:
                            tp += 1
                        else:
                            fp += 1
                    elif is_point_in_image(x_pred, y_pred) and not is_point_in_image(x_gt, y_gt):
                        fp += 1
                    elif not is_point_in_image(x_pred, y_pred) and is_point_in_image(x_gt, y_gt):
                        fn += 1
                    elif not is_point_in_image(x_pred, y_pred) and not is_point_in_image(x_gt, y_gt):
                        tn += 1

            eps = 1e-15
            precision = round(tp / (tp + fp + eps), 5)
            accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
            print('val, epoch = {}, iter_id = {}/{}, loss = {}, tp = {}, fp = {}, fn = {}, tn = {}, precision = {}, '
                  'accuracy = {}'.format(epoch, iter_id, len(val_loader), round(loss.item(), 5), tp, fp, fn, tn,
                                         precision, accuracy))
            losses.append(loss.item())
    return np.mean(losses), tp, fp, fn, tn, precision, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to pretrained model')
    args = parser.parse_args()

    val_dataset = courtDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    criterion = nn.MSELoss()

    val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, -1)





