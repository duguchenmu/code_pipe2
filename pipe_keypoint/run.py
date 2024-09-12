import argparse
import os
from os.path import join
import torch
import cv2
from matplotlib import pyplot as plt
from pipe_keypoint.models.point_pipe import TwoViewPipePoint
from pipe_keypoint import numpy_image_to_torch, batch_to_np
from pipe_keypoint.drawing import plot_keypoints, plot_images

def main():
    parser = argparse.ArgumentParser(
        prog='Superpoint_test'
    )
    parser.add_argument('-img1', default=join('E:\\code_pipei\\new_pipei\\1.jpg'))
    parser.add_argument('-img2', default=join('E:\\code_pipei\\new_pipei\\2.png'))
    parser.add_argument('--max_pts', type=int, default=500)
    args = parser.parse_args()

    conf = {
        'name': 'point_pipe',
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipepoint_model = TwoViewPipePoint(conf).to(device).eval()

    gray0 = cv2.imread(args.img1, 0)
    from picture_process import extract_screen, edgepoint_elimination, dbscan_cluster, update_pred
    gray0 = extract_screen(gray0)
    # cv2.namedWindow('Screen', 0)
    # cv2.imshow("Screen", gray0)
    # cv2.imwrite("gray0.png", gray0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    gray1 = cv2.imread(args.img2, 0)

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}
    pred = pipepoint_model(x)

    pred = batch_to_np(pred)
    kp0, kp1 = pred['keypoints0'], pred['keypoints1']
    kp0 = edgepoint_elimination(gray0, kp0)
    print(kp0)
    indices_ = dbscan_cluster(kp0)
    pred_new = update_pred(pred, indices_)

    print(pred_new)
    # print(kp0)
    # print(kp1)
    # print(type(kp0))
    # print(type(kp1))
    import csv
    with open('kp0.csv', 'w', newline='') as file_obj:
        writer = csv.writer(file_obj)
        for k in kp0:
            writer.writerow(k)
    with open('kp1.csv', 'w', newline='') as file_obj:
        writer = csv.writer(file_obj)
        for k in kp1:
            writer.writerow(k)

    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    plot_images([img0, img1], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
    plot_keypoints([kp0, kp1], colors='c')
    plt.gcf().canvas.manager.set_window_title('Detected Points')
    plt.savefig('detected_points00.png')

if __name__ == '__main__':
    main()