import numpy as np
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import transform


# 7-segment display mapping
DIGITS_LOOKUP = {
    0: (1, 1, 1, 0, 1, 1, 1),
    1: (0, 0, 1, 0, 0, 1, 0),
    2: (1, 0, 1, 1, 1, 0, 1),
    3: (1, 0, 1, 1, 0, 1, 1),
    4: (0, 1, 1, 1, 0, 1, 0),
    5: (1, 1, 0, 1, 0, 1, 1),
    6: (1, 1, 0, 1, 1, 1, 1),
    7: (1, 0, 1, 0, 0, 1, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1)
}


def read_digits_screen(img, n_clusters, color, w_crop_corr, h_crop_corr, n_digits, delta_w, segment_width):

    # Convert to RGB representation
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Rotate image (degrees)
    img = ndimage.rotate(img, -3.5)

    # Crop image (grossly)
    img = img[120:240, 1029:1252]

    # K-means clustering
    img_clustered, centroids, labels = cluster_image(img, n_clusters)

    # Crop image
    img_cropped = crop_image(img, n_clusters, centroids, labels, color, w_crop_corr, h_crop_corr)

    # Crop single digits
    imgs_single_digit = crop_single_digits(img_cropped, n_digits, delta_w)

    # Build digit masks
    masks = build_digit_masks(imgs_single_digit, segment_width)

    # Classify digits and obtain reading on the number on the screen
    reading = classify_digits(masks, imgs_single_digit, n_digits)

    return reading


def cluster_image(img, n_clusters):
    # Reshape and convert to np.float32
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    # Define criteria and apply K-means
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv.kmeans(Z, n_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Convert back into uint8, and make original image
    center = np.uint8(center)
    img_clustered = center[label.flatten()]
    img_clustered = img_clustered.reshape((img.shape))

    # Display clustered image
    cv.imshow('clustered (K-means)', img_clustered)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_clustered, center, label


def crop_image(img, n_clusters, centroids, labels, color, w_correction, h_correction):

    # Extract binary mask to only keep digits
    idx_digits = 0
    min_dist = np.Inf
    for i in range(n_clusters):
        dist = np.linalg.norm(np.subtract(np.asarray(centroids[i]), color))
        if dist < min_dist:
            min_dist = dist
            idx_digits = i

    idx_other_classes = list(range(n_clusters))
    idx_other_classes.pop(idx_digits)

    centroids[idx_other_classes,:] = np.uint8([0, 0, 0])
    centroids[idx_digits,:] = np.uint8([255, 255, 255])

    labels = labels.flatten()
    digit_mask = centroids[labels]
    digit_mask = digit_mask.reshape((img.shape))
    digit_mask = cv.cvtColor(digit_mask, cv.COLOR_RGB2GRAY)

    # Display digit mask
    cv.imshow('digit mask', digit_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Crop image to select only the screen region with the digits
    whites_h = np.sort(np.where(digit_mask[:,:] == 255)[0])
    whites_h_bounds = list((whites_h[0], whites_h[-1]))
    whites_w = np.sort(np.where(digit_mask[:,:] == 255)[1])
    whites_w_bounds = list((whites_w[0], whites_w[-1]))

    img_cropped = digit_mask[whites_h_bounds[0]+h_correction:whites_h_bounds[1],
                             whites_w_bounds[0]+w_correction:whites_w_bounds[1]]

    # Display cropped image
    cv.imshow('cropped', img_cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_cropped


def crop_single_digits(img, n_digits, delta_w):
    # Copy image to highlight digit regions
    img_ROI = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_ROI_copy = img_ROI.copy()

    # Define crop width and height
    bbox_w = int((img.shape[1] - (n_digits-1) * delta_w) / n_digits)
    up_h = 0
    bottom_h = img.shape[0]

    # List to store all digit regions
    imgs_single_digit = []

    # Scan the image horizontally from left to right
    w = img.shape[1]

    while w >= bbox_w:
        # Update sliding window
        up_w = w - bbox_w
        bottom_w = w
        w -= (bbox_w + delta_w)

        # Draw ROI rectangle on image
        cv.rectangle(img_ROI, (up_w, up_h), (bottom_w, bottom_h), (0, 0, 255), 1)
        
        # Select and store the region with the single digit
        single_digit = img_ROI_copy[up_h:bottom_h, up_w:bottom_w, :]

        # # Add rotation
        # digit_size = single_digit.shape
        # single_digit = ndimage.rotate(single_digit, 5)
        #
        # # Resize image after cropping
        # single_digit = single_digit[:, 4:single_digit.shape[1]-3, :]
        # cv.resize(single_digit, (digit_size[0], digit_size[1]))

        cv.imshow('single digit', single_digit)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # # Warp image
        # pts1 = np.float32([[8, 10], [single_digit.shape[0], 3], [8, 43], [single_digit.shape[0], 38]])
        # pts2 = np.float32([[0, 0], [single_digit.shape[0], 0], [0, single_digit.shape[1]],
        #                    [single_digit.shape[0], single_digit.shape[1]]])
        # M = cv.getPerspectiveTransform(pts1, pts2)
        # single_digit = cv.warpPerspective(single_digit, M, (single_digit.shape[1], single_digit.shape[0]))
        #
        # color1 = (0, 255, 0)
        # color2 = (255, 0, 0)
        # markerType = cv.MARKER_CROSS
        # markerSize = 15
        # thickness = 2
        # for point in pts1:
        #     cv.drawMarker(single_digit, (point[0], point[1]), color1, markerType, markerSize, thickness)
        # for point in pts2:
        #     cv.drawMarker(single_digit, (point[0], point[1]), color2, markerType, markerSize, thickness)

        cv.imshow('single digit', single_digit)
        cv.waitKey(0)
        cv.destroyAllWindows()

        imgs_single_digit.append(single_digit)

    # Visualize the cropped image with all digit regions highlighted
    cv.imshow('img with digit ROI', img_ROI)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return imgs_single_digit


def build_digit_masks(imgs_single_digit, segment_width):
    # Initialize mask
    mask = np.zeros(imgs_single_digit[0].shape, dtype = np.uint8)

    # Define points to draw segments
    w1 = segment_width
    w2 = mask.shape[1] - w1
    h1 = w1
    h2 = int(mask.shape[0] / 2 - w1 / 2)
    h3 = h2 + w1
    h4 = mask.shape[0] - w1

    # Associate the coordinate of each point to the corresponding segment index 
    # - 0 -
    # 1   2
    # - 3 -
    # 4   5
    # - 6 -
    positions_lookup = {
        0: [(0, 0), (mask.shape[1], h1)],
        1: [(0, 0), (w1, h3)],
        2: [(w2, 0), (mask.shape[1], h3)],
        3: [(0, h2), (mask.shape[1], h3)],
        4: [(0, h2), (w1, mask.shape[0])],
        5: [(w2, h2), (mask.shape[1], mask.shape[0])],
        6: [(0, h4), (mask.shape[1], mask.shape[0])]
    }

    # Create a mask for each digit
    masks = []
    for digit in range(10):
        mask = np.zeros(imgs_single_digit[0].shape, dtype = np.uint8)
        for i in range(7):
            if DIGITS_LOOKUP[digit][i] == 1:
                # Draw segment
                cv.rectangle(mask, positions_lookup[i][0], positions_lookup[i][1], (255, 255, 255), -1)

                # # Display segment on mask (incremental)
                # cv.imshow('mask', mask)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
        
        # # Display whole mask
        # cv.imshow('mask', mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        masks.append(mask)

    return masks


# LIMITATIONS: only positive numbers in [0,10)
def classify_digits(masks, imgs_single_digit, n_digits):
    detected_digits = []

    # Classify each digit
    for i in range (n_digits):
        # Detect which ideal mask is most similar to the cropped image mask
        max_similarity = 0
        idx_max = 0

        print(f'Digit in position {i} from the right of the screen')
        for mask in range(10):
            # # Display ideal mask
            # cv.imshow('mask', masks[mask])
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # # Display cropped image mask
            # cv.imshow('actual digit', imgs_single_digit[i])
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            similarity = np.sum(imgs_single_digit[i] == masks[mask])
            print(f'Current mask: {mask} | Similarity: {similarity}')

            if similarity > max_similarity:
                idx_max = mask
                max_similarity = similarity

        # print(f'Detected digit: {idx_max}\n')
        detected_digits.append(idx_max)

    # Reconstruct number from digits
    multiplier = 1e-3
    reading = 0
    for i in range(n_digits):
        reading += detected_digits[i]*multiplier
        multiplier *= 10

    reading = np.round(reading, decimals=3)
    # print(f'Screen reading: {reading}')

    return reading

    
def main():
    # Define parameters
    n_clusters = 6                      # number of clusters for the k-means algorithm
    color = np.asarray([250, 250, 230])  # color of the selected cluster (digits)
    w_crop_corr = 15                    # screen crop to select only the digits (width)
    h_crop_corr = 20                    # screen crop to select only the digits (height)
    delta_w = 6                         # horizontal space between digits
    n_digits = 4                        # number of digits on the screen
    segment_width = 7                   # thickness of each digit segment

    # Read image
    img = cv.imread('./img_digits/frame_1.png')
    # img = cv.rotate(img, cv.ROTATE_180)
    # img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    # Display original image 
    cv.imshow('original image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Read number on the screen (classification using similarity to digit masks)
    reading = read_digits_screen(img, n_clusters, color, w_crop_corr, h_crop_corr, n_digits, delta_w, segment_width)
    print(f'Number read on the screen: {reading}')

    # Display original image and overlay reading
    cv.putText(img, str(reading), org=(img.shape[1]-250,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
               fontScale=2, color=(255, 0, 0), thickness=8, lineType=cv.LINE_AA)
    cv.imshow('original image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


