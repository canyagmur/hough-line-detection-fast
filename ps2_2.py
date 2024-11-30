import cv2
import numpy as np
import matplotlib.pyplot as plt
from HoughTransform import HoughTransform
import time





    

def main():
    # Create an instance of the HoughTransform class
    image_width,image_height = 640,480
    ht = HoughTransform(image_shape=(image_height,image_width),use_dilete_and_erode=True)

    # Define the video capture device
    cap = cv2.VideoCapture(0)

    # Set the video capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    # Initialize FPS variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        # Read a frame from the video capture device
        ret, frame = cap.read()

        # Compute the accumulator matrix using the edges
        H = ht.hough_lines_acc_vectorized(frame)

        # Find the peaks in the accumulator matrix
        peaks = ht.hough_peaks_vectorized(H, count_threshold=100 ,take_strongest_n=5,apply_nms=True,nms_threshold=0.05)

        # Draw lines on the original frame corresponding to the peaks
        img = ht.hough_lines_draw(frame, peaks)



        # Increment frame count and calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Add FPS overlay on image
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Hough Lines Demo', img)

        # Wait for key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()