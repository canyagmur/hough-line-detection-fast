import cv2
import numpy as np
import time
import argparse
import sys

def main(path_to_image,count_threshold):
    # Load the image and convert it to grayscale
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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 1)
    # Apply edge detection to the image
    edges = cv2.Canny(gray, 50, 150)

    # Perform the Hough Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=count_threshold)

    # Draw the detected lines on a copy of the original image
    lines_img = image.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print("--- OpenCV  %s : %s seconds ---" % (image_name,time.time() - start_time))

    # Show the original image, the edge image, and the lines image
    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Image', edges)
    cv2.imshow('Hough Lines', lines_img)
    cv2.imwrite('output/opencv_hough_{}.png'.format(image_name), lines_img)

    # Wait for a key press and then close all windows
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
