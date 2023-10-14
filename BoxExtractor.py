import cv2
import numpy as np
import os

class ImageProcessor:
    def __init__(self):
        self.output_folder = 'cropped_boxes'
        self.img_output = 'contour_images'

    def box_extraction(self, img_for_box_extraction_path):
        img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
        (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
        img_bin = 255 - img_bin  # Invert the image

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Defining a kernel length
        kernel_length = np.array(img).shape[1] // 40

        # A verticle kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal lines from the image.
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Morphological operation to detect vertical lines from an image
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
        self.store_process_image("1_vertical_img_lines.jpg", verticle_lines_img)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

        self.store_process_image("2_horizontal_img_lines.jpg", horizontal_lines_img)

        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha
        # This function helps to add two images with specific weight parameters to get a third image as the summation of two images.
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)



        # For Debugging
        # Enable this line to see vertical and horizontal lines in the image which are used to find boxes
        # cv2.imwrite(os.path.join(self.output_folder, "img_final_bin.jpg"), img_final_bin)

        self.store_process_image("3_img_final_bin.jpg", img_final_bin)

        # Find contours for the image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort all the contours by top to bottom.
        (contours, boundingBoxes) = self.sort_contours(contours, method="top-to-bottom")
        idx = 0

        for c in contours:
            # Returns the location and width, height for every contour
            x, y, w, h = cv2.boundingRect(c)
            # If the box height is greater than 20, width is > 80, then only save it as a box in the "cropped/" folder.
            if (w > 80 and h > 20) and w > 3 * h:
                idx += 1
                new_img = img[y:y+h, x:x+w]
                self.store_boxes(idx, new_img)
                

    @staticmethod
    def sort_contours(cnts, method="left-to-right"):
        # Initialize the reverse flag and sort index
        reverse = False
        i = 0
        # Handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # Handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # Construct the list of bounding boxes and sort them from top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)
    
    def store_process_image(self, file_name, image):

        if not os.path.exists(self.img_output):
            os.makedirs(self.img_output)

        cv2.imwrite(os.path.join(self.img_output, file_name), image)

    def store_boxes(self, idx, image):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        cv2.imwrite(os.path.join(self.output_folder, str(idx) + '.png'), image)

