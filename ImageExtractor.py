# image_to_pdf.py

import os
from pdf2image import convert_from_path

class PDFToImagesConverter:
    def __init__(self, pdf_file, output_folder):
        self.pdf_file = pdf_file
        self.output_folder = output_folder

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def convert_to_images(self):
        # Convert the PDF to a list of images
        images = convert_from_path(self.pdf_file)

        # Loop through the list of images and save each one in the output folder
        for i, image in enumerate(images):
            image.save(os.path.join(self.output_folder, f"page_{i + 1}.jpg"), "JPEG")
