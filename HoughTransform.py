import cv2
import numpy as np
import matplotlib.pyplot as plt

import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds to complete.")
        return result
    return wrapper



class HoughTransform:
    def __init__(self,image_shape=None,theta_resolution=1,rho_resolution=1,use_dilete_and_erode=False) -> None:
        self.theta_resolution = theta_resolution
        self.rho_resolution = rho_resolution
        self.use_dilete_and_erode = use_dilete_and_erode
        self.height, self.width = image_shape[:2] #look again
        self.diagonal_distance = int(np.ceil(np.sqrt(self.height**2 + self.width**2)))
        self.theta_range = np.arange(0, 180, self.theta_resolution)
        self.rho_range = np.arange(-self.diagonal_distance, self.diagonal_distance, self.rho_resolution)
        self.accumulator = np.zeros((len(self.rho_range), len(self.theta_range)), dtype=np.uint64)
        #self.edge_image = self.prepare_edge_image(image,self.use_dilete_and_erode)
        self.cos_thetas = np.cos(np.deg2rad(self.theta_range))
        self.sin_thetas = np.sin(np.deg2rad(self.theta_range))


    # def init_parameters(self,image):
    #     self.image = image
    #     self.height, self.width = image.shape[:2] #look again
    #     self.diagonal_distance = int(np.ceil(np.sqrt(self.height**2 + self.width**2)))
    #     self.theta_range = np.arange(0, 180, self.theta_resolution)
    #     self.rho_range = np.arange(-self.diagonal_distance, self.diagonal_distance, self.rho_resolution)
    #     self.accumulator = np.zeros((len(self.rho_range), len(self.theta_range)), dtype=np.uint64)
    #     self.edge_image = self.prepare_edge_image(image,self.use_dilete_and_erode)
    #     self.cos_thetas = np.cos(np.deg2rad(self.theta_range))
    #     self.sin_thetas = np.sin(np.deg2rad(self.theta_range))

    #@measure_time
    def prepare_edge_image(self,image,use_dilete_and_erode=False):
            edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
            edge_image = cv2.Canny(edge_image, 50, 150)
            if use_dilete_and_erode:
                edge_image = cv2.dilate(
                    edge_image,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                    iterations=1
                )
                edge_image = cv2.erode(
                    edge_image,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                    iterations=1
                )
            return edge_image
    
    #@measure_time
    def hough_lines_acc(self,image):
        """ Takes an edge image as input,
            Returns an accumulator matrix H, and cell locations theta and rho
        """
        edge_image = self.prepare_edge_image(image,self.use_dilete_and_erode)
        for i in range(self.height):
            for j in range(self.width):
                if edge_image[i,j] > 0:
                    for theta_idx in range(len(self.theta_range)):
                        #theta = np.deg2rad(self.theta_range[theta_idx])
                        rho = int(np.round(j*self.cos_thetas[theta_idx] + i*self.sin_thetas[theta_idx])+self.diagonal_distance)
                        #rho_idx = np.argmin(np.abs(self.rho_range - rho))
                        self.accumulator[rho, theta_idx] += 1
        
        return self.accumulator
    
    #@measure_time
    def hough_lines_acc_vectorized(self,image):
        """ Takes an edge image as input,
            Returns an accumulator matrix H, and cell locations theta and rho
        """
        # Compute rho values for all edge pixels and all possible values of theta

        edge_image = self.prepare_edge_image(image,self.use_dilete_and_erode)

        edge_points = np.argwhere(edge_image != 0)

        rho_values = np.matmul(edge_points, np.array([self.sin_thetas, self.cos_thetas]))

        self.accumulator, theta_vals, rho_vals = np.histogram2d(
            np.tile(self.theta_range, rho_values.shape[0]),
            rho_values.ravel(),
            bins=[self.theta_range, self.rho_range]
        )
        self.accumulator = np.transpose(self.accumulator)
        
        return self.accumulator
    

    def non_max_suppression(self,peaks,accumulator_shape, nms_threshold=0.3):
        """ Performs non-maximum suppression on the accumulator array
        """
        # convert peaks to numpy array
        peaks = np.array(peaks)

        # initialize list of selected peaks
        selected_peaks = []

        while peaks.size > 0:
            # select the peak with the highest score
            current_peak = peaks[0]
            selected_peaks.append(current_peak)

            # calculate the distance between the current peak and all other peaks
            distances = np.sqrt(np.sum((peaks[:, :2] - current_peak[:2]) ** 2, axis=1))

            # select the peaks that are far enough from the current peak
            mask = distances > nms_threshold * max(accumulator_shape)
            #print(mask)
            peaks = peaks[mask]

        # convert selected peaks back to list of tuple
        peaks = [(int(peak[0]),int(peak[1])) for peak in selected_peaks]        

        return peaks



    #@measure_time
    def hough_peaks(self,H,count_threshold,take_strongest_n=0,apply_nms=False, nms_threshold=0.3):
        """"
         Returns the peak locations (theta and rho) for a given H, theta, and rho
        arrays. The function also should take a threshold value t to eliminate weak peaks that
        are less than t, and another parameter s to return the strongest s peaks. 
        """
        peaks = []
        for i in range(len(H)):
            for j in range(len(H[0])):
                if H[i,j] > count_threshold:
                    peaks.append({(i,j) : H[i,j]})
        #sort peaks by value
        #print("before : ",peaks[:3])
        peaks = sorted(peaks, key=lambda x: list(x.values())[0], reverse=True)

        #print("after : ",peaks[:3])

        if apply_nms:
            peaks = self.non_max_suppression(peaks,accumulator_shape=H.shape,nms_threshold=nms_threshold)

        if take_strongest_n:
            peaks = peaks[:take_strongest_n]


        #only return the keys
        peaks = [list(peak.keys())[0] for peak in peaks]
        return peaks
    
    
    #@measure_time
    def hough_peaks_vectorized(self,H,count_threshold,take_strongest_n=0,apply_nms=False, nms_threshold=0.3):
        """"
         Returns the peak locations (theta and rho) for a given H, theta, and rho
        arrays. The function also should take a threshold value t to eliminate weak peaks that
        are less than t, and another parameter s to return the strongest s peaks. 
        """
        line_indices = np.argwhere(H > count_threshold)
        # create dictionary of index-value pairs
        peaks = {(r, c): H[r, c] for r, c in line_indices}
        peaks = sorted(peaks, key=lambda x: peaks[x], reverse=True)
        #print("after : ",peaks[:3])

        if apply_nms:
            peaks = self.non_max_suppression(peaks,accumulator_shape=H.shape,nms_threshold=nms_threshold)   

        if take_strongest_n:
            peaks = peaks[:take_strongest_n]
        
        #only return the keys
        peaks = list(peaks)
        peaks = [list(peak) for peak in peaks]
        

        return peaks    
    
    #@measure_time
    def hough_lines_draw(self,img,peaks):
        """  Draw color lines corresponding to the peaks found.
        """
        for peak in peaks:
            rho = self.rho_range[peak[0]]
            theta = self.theta_range[peak[1]]
            a = np.cos(np.deg2rad(theta)) 
            b = np.sin(np.deg2rad(theta)) 
            x0 = a*rho #
            y0 = b*rho #
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,255,255),2)
        return img
