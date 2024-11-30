import cv2
import numpy as np
import matplotlib.pyplot as plt
from HoughTransform import HoughTransform
import time
import argparse
import sys







    

def main(path_to_image,count_threshold):

    start_time = time.time()

    # Read the image file
    try:
        image = cv2.imread(path_to_image)
        if image is None:
            raise ValueError('Invalid image file')
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
    
    image_name = path_to_image.split('/')[-1][:-4]

    # Create an instance of the HoughTransform class
    ht = HoughTransform(image_shape=image.shape, use_dilete_and_erode=False)

    # Compute the accumulator matrix
    H = ht.hough_lines_acc_vectorized(image) #ht.hough_lines_acc()

    # # Create 1D arrays for the x and y coordinates
    # x = np.arange(H.shape[1])
    # y = np.arange(H.shape[0])

    # # Use meshgrid to create a 2D coordinate grid
    # X, Y = np.meshgrid(x, y)

    # # Find the indices of the maximum values
    # max_indices = np.argwhere(H > np.max(H)*0.3)
    # # Plot a star at the location of the maximum values
    # for idx in max_indices:
    #     plt.plot(idx[1] + 0.5, idx[0] + 0.5, '*', color='red', markersize=5)

    # # Plot the array values as a heatmap
    # plt.pcolormesh(X, Y, H, cmap='viridis')
    # plt.colorbar()
    # plt.show()


    #Find the peaks in the accumulator matrix
    peaks = ht.hough_peaks_vectorized(H, count_threshold=count_threshold, take_strongest_n=0,apply_nms=True,nms_threshold=0.05) #ht.hough_peaks_vectorized(H, count_threshold=H.max() *0.3, take_strongest_n=20)
    img = ht.hough_lines_draw(image.copy(), peaks)

    print("--- Can  %s : %s seconds ---" % (image_name,time.time() - start_time))


    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Image', ht.prepare_edge_image(image))
    cv2.imshow('Hough Lines', img)
    cv2.imwrite('output/can_hough_{}.png'.format(image_name), img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process an image file.')
    parser.add_argument('--path_to_image', help='Path to the image file to process', required=True)
    parser.add_argument('--count_threshold', type=int, help='Count threshold for processing the image', required=True)

    # Parse the arguments
    args = parser.parse_args()
    main(path_to_image=args.path_to_image,count_threshold=args.count_threshold)