import cv2
import numpy as np
import time

def main():
    # Define the video capture device
    cap = cv2.VideoCapture(0)

    # Set the video capture properties
    image_width, image_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    # Initialize FPS variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        # Read a frame from the video capture device
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Perform Hough line transform
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

        # Draw the detected lines on the frame
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Increment frame count and calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Add FPS overlay on image
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Hough Lines Demo', frame)

        # Wait for key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
