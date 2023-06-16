import numpy as np
import cv2
import os
from utils import is_point_in_image
from scipy.spatial import distance
from postprocess import refine_kps
from homography import get_trans_matrix, refer_kps
from postprocess import postprocess
from dataset import courtDataset
from tracknet import BallTrackerNet
import argparse
import torch
import torch.nn.functional as F

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
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
    device = 'cuda'
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    tp, tn, fp, fn = 0, 0, 0, 0
    MAX_DIST = 7
    dists = []
    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            batch_size = batch[0].shape[0]
            out = model(batch[0].float().to(device))
            kps = batch[2]
            img_names = batch[3]
            gt_hm = batch[1].float().to(device)

            pred = F.sigmoid(out).detach().cpu().numpy()
            for bs in range(batch_size):
                img = cv2.imread(os.path.join(val_dataset.path_dataset, 'images', img_names[bs] + '.png'))
                points_pred = []
                for kps_num in range(14):
                    heatmap = (pred[bs][kps_num] * 255).astype(np.uint8)
                    x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
                    if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                        x_pred, y_pred = refine_kps(img, int(y_pred), int(x_pred))
                    points_pred.append((x_pred, y_pred))

                if args.use_homography:
                    matrix_trans = get_trans_matrix(points_pred)
                    if matrix_trans is not None:
                        points_pred = cv2.perspectiveTransform(refer_kps, matrix_trans)
                        points_pred = [np.squeeze(x) for x in points_pred]

                for i, point_pred in enumerate(points_pred):
                    x_gt = kps[bs][i][0]
                    y_gt = kps[bs][i][1]
                    x_pred = point_pred[0]
                    y_pred = point_pred[1]

                    if is_point_in_image(x_pred, y_pred) and is_point_in_image(x_gt, y_gt):
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        dists.append(dst)
                        if dst < MAX_DIST:
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
                precision = round(tp/(tp+fp+eps), 5)
                accuracy = round((tp+tn)/(tp+tn+fp+fn+eps), 5)
                print('i = {}, tp = {}, fp = {}, fn = {}, tn = {}, prec = {}, acc = {}, mean_dist = {}'.format(iter_id, tp, fp, fn, tn, precision, accuracy, np.median(dists)))
    
    
    
    
    