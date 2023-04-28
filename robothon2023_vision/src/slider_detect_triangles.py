import cv2
import numpy as np
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
from math import sin, cos
import random as rng
import os


def crop_image(img, lb, ub):
    # Select screen background
    lower_color = np.array(lb, dtype = "uint16")
    upper_color = np.array(ub, dtype = "uint16")
    screen_mask = cv.inRange(img, lower_color, upper_color)
    # cv2.imwrite(path,screen_mask)
    # plt.imshow(screen_mask, cmap=plt.cm.gray), plt.show()
    # cv2.imshow("screen mask", screen_mask),
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    failed = False
    if len(np.where(screen_mask[:,:] == 255)[0]) == 0:
        print("No screen detected")
        failed = True
        return img, None, None, failed

    whites_h = np.sort(np.where(screen_mask[:,:] == 255)[0])
    whites_h_bounds = list((whites_h[0], whites_h[-1]))
    whites_w = np.sort(np.where(screen_mask[:,:] == 255)[1])
    whites_w_bounds = list((whites_w[0], whites_w[-1]))

    reduction_w_left = 4
    reduction_w_right = 4
    reduction_h_head = 3
    ROI_h = np.min([55,whites_h_bounds[1]-whites_h_bounds[0]])
    img_cropped = img[whites_h_bounds[0]+reduction_h_head:whites_h_bounds[0]+ROI_h,
                      whites_w_bounds[0]+reduction_w_left:whites_w_bounds[1]-reduction_w_right]
    # cv2.imshow("img_cropped", img_cropped),
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    # Show cropped image
    # plt.imshow(img_cropped), plt.title("Cropped to screen"), plt.show()

    return img_cropped, whites_h_bounds[0]+reduction_h_head, whites_w_bounds[0]+reduction_w_left, failed


def preprocess_image_auto(img, lb, ub):

    # Crop image (only screen)
    img, lb_h, lb_w, failed = crop_image(img, lb, ub)

    if failed:
        return None, None, None, failed

    # Select and mask off screen background
    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([180,255,255], dtype = "uint16")
    screen_mask = cv.bitwise_not(cv.inRange(img, lower_black, upper_black))
    # plt.imshow(screen_mask, cmap=plt.cm.gray), plt.title("Screen masked off"), plt.show()
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    # Canny Edge Detection
    # screen_mask = cv.GaussianBlur(img, (5,5), 0) 
    edges = cv.Canny(image=screen_mask, threshold1=100, threshold2=250)

    # plt.imshow(edges, cmap=plt.cm.gray), plt.show()

    return screen_mask, edges, (lb_h, lb_w), failed


def find_centroids_independent(img, dist_percent, h_triangle, w_triangle):
    img_final = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    actual_1 = actual_2 = target_1 = target_2 = None

    # Select midpoint 
    w_mid = int(img.shape[1] / 2)
    dw = 25
    w_mid_range = [w_mid - dw, w_mid + dw]

    # Find first white from top
    if not np.where(img[:,w_mid_range[0]:w_mid_range[1]])[0].any():
        print("No triangles detected")
        return img_final, actual_1, actual_2, target_1, target_2
    
    first_white_in_mid_range = np.where(img[:,w_mid_range[0]:w_mid_range[1]])[0][0]

    # Determine height of the cutting lines based on the first white detected
    h1 = 5 + first_white_in_mid_range
    h2 = h1 + int(1.0*h_triangle)
    
    h1_b = h1+int(0.01*dist_percent*h_triangle)
    h2_b = h2+int(0.01*dist_percent*h_triangle)

    cv.line(img_final, (0, h1), (img.shape[1], h1), (0,255,0), 1, cv.LINE_AA)
    cv.line(img_final, (0, h2), (img.shape[1], h2), (0,255,0), 1, cv.LINE_AA)
    cv.line(img_final, (0, h1_b), (img.shape[1], h1_b), (0,255,0), 1, cv.LINE_AA)
    cv.line(img_final, (0, h2_b), (img.shape[1], h2_b), (0,255,0), 1, cv.LINE_AA)

    # Actual
    two_actual_detected = False
    zero_actual_detected = False

    sum_up_fronts = -1
    while (not(sum_up_fronts in [0, 2, 4]) and h2 <= h2_b):
        line = img[h2,:]
        line_diff = np.ediff1d(line, to_begin=line[0])
        up_fronts = line_diff == 255
        idx_up_fronts = np.where(up_fronts)[0]
        sum_up_fronts = np.sum(up_fronts)
        h2 += 1

    if sum_up_fronts == 2:
        if idx_up_fronts[1] - idx_up_fronts[0] > int(1.5*w_triangle):
            print("2 actual detected (overlap)")
            x_centroid_actual_1 = int(idx_up_fronts[0] + int(0.5*w_triangle))
            x_centroid_actual_2 = int(idx_up_fronts[1] - int(0.5*w_triangle))
            two_actual_detected = True
        else:
            print("1 actual detected")
            x_centroid_actual_1 = int((idx_up_fronts[0] + idx_up_fronts[1]) / 2)
            x_centroid_actual_2 = None

    elif sum_up_fronts == 4:
        print("2 actual detected (no overlap)")
        x_centroid_actual_1 = int((idx_up_fronts[0] + idx_up_fronts[1]) / 2)
        x_centroid_actual_2 = int((idx_up_fronts[2] + idx_up_fronts[3]) / 2)
        two_actual_detected = True

    else:
        print("No actual detected")
        x_centroid_actual_1 = None
        x_centroid_actual_2 = None
        zero_actual_detected = True

    y_centroid_actual_1 = int((h2 + h2_b) / 2.0)
    y_centroid_actual_2 = y_centroid_actual_1


    # Targets
    two_targets_detected = False

    sum_up_fronts = -1
    while (not(sum_up_fronts in [0, 2, 4]) and h1 <= h1_b):
        line = img[h1,:]
        line_diff = np.ediff1d(line, to_begin=line[0])
        up_fronts = line_diff == 255
        idx_up_fronts = np.where(up_fronts)[0]
        sum_up_fronts = np.sum(up_fronts)
        h1 += 1

    if sum_up_fronts == 2:
        if idx_up_fronts[1] - idx_up_fronts[0] > int(1.5*w_triangle):
            print("2 targets detected (overlap)")
            x_centroid_target_1 = int(idx_up_fronts[0] + int(0.5*w_triangle))
            x_centroid_target_2 = int(idx_up_fronts[1] - int(0.5*w_triangle))
            two_targets_detected = True
        else:
            print("1 target detected")
            x_centroid_target_1 = int((idx_up_fronts[0] + idx_up_fronts[1]) / 2)
            x_centroid_target_2 = None

    elif sum_up_fronts == 4:
        print("2 targets detected (no overlap)")
        x_centroid_target_1 = int((idx_up_fronts[0] + idx_up_fronts[1]) / 2)
        x_centroid_target_2 = int((idx_up_fronts[2] + idx_up_fronts[3]) / 2)
        two_targets_detected = True

    else:
        print("No targets detected")
        x_centroid_target_1 = None
        x_centroid_target_2 = None

    y_centroid_target_1 = int((h1 + h1_b) / 2.0)
    y_centroid_target_2 = y_centroid_target_1


    # Swap target 1 and target 2 based on proximity to midpoint of the image
    if two_targets_detected and (abs(x_centroid_target_1-w_mid) > abs(x_centroid_target_2-w_mid)):
        tmp = x_centroid_target_1
        x_centroid_target_1 = x_centroid_target_2
        x_centroid_target_2 = tmp

    # Save detected centroids in tuples and color them in the binary image
    if x_centroid_actual_1:
        actual_1 = (y_centroid_actual_1, x_centroid_actual_1)
        img_final[actual_1[0], actual_1[1], 0] = 255
    if x_centroid_actual_2:
        actual_2 = (y_centroid_actual_2, x_centroid_actual_2)
        img_final[actual_2[0], actual_2[1], 0] = 255
    if x_centroid_target_1:
        target_1 = (y_centroid_target_1, x_centroid_target_1)
        img_final[target_1[0], target_1[1], 0] = 255
    if x_centroid_target_2:
        target_2 = (y_centroid_target_2, x_centroid_target_2)    
        img_final[target_2[0], target_2[1], 0] = 255

    # plt.imshow(img_final), plt.title("Centroid detection on mask"), plt.show()
    
    return img_final, actual_1, actual_2, target_1, target_2


# def main():
#     lb_color = [0,230,245]
#     ub_color = [120,255,255]
#
#     dist_percent = 20
#     h_triangle = 20
#     w_triangle = 18
#
#     path = 'img_raw'
#     abs_path = os.path.join(os.getcwd(),path)
#
#     dir_list = os.listdir(abs_path)
#
#     for image_n in range(0,100):
#         try:
#             img = np.asarray(Image.open(abs_path + '/' + dir_list[image_n]))
#             # img = np.asarray(Image.open(abs_path + '/' + 'screen_230420_113624.png'))
#         except IOError:
#             pass
#
#         print('\n' + dir_list[image_n])
#         # plt.imshow(img), plt.title("Original image"), plt.show()
#
#         # Preprocess image
#         screen_mask, edges, bounds, failed = preprocess_image_auto(img, lb_color, ub_color)
#
#         if failed:
#             plt.imshow(img), plt.title("Centroid detection on original image"), plt.show()
#             break
#
#         # Find centroids
#         img_final, actual_1, actual_2, target_1, target_2 = find_centroids_independent(edges, dist_percent, h_triangle, w_triangle)
#
#         # Overlay centroids on original image
#         if target_1:
#             h_target_1 = target_1[1] + bounds[1]
#             w_target_1 = target_1[0] + bounds[0]
#             cv.drawMarker(img, (h_target_1, w_target_1), (0, 200, 0), cv.MARKER_TRIANGLE_DOWN, 6, 3)
#
#         if target_2:
#             h_target_2 = target_2[1] + bounds[1]
#             w_target_2 = target_2[0] + bounds[0]
#             cv.drawMarker(img, (h_target_2, w_target_2), (100, 100, 0), cv.MARKER_TRIANGLE_DOWN, 6, 3)
#
#         if actual_1:
#             h_actual_1 = actual_1[1] + bounds[1]
#             w_actual_1 = actual_1[0] + bounds[0]
#             cv.drawMarker(img, (h_actual_1, w_actual_1), (0, 0, 0), cv.MARKER_TRIANGLE_UP, 6, 3)
#
#         if actual_2:
#             h_actual_2 = actual_2[1] + bounds[1]
#             w_actual_2 = actual_2[0] + bounds[0]
#             cv.drawMarker(img, (h_actual_2, w_actual_2), (0, 100, 100), cv.MARKER_TRIANGLE_UP, 6, 3)
#
#         plt.imshow(img), plt.title("Centroid detection on original image"), plt.show()

#
# if __name__ == "__main__":
#    main()