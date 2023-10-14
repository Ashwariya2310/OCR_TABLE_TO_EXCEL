import cv2
from ImageExtractor import PDFToImagesConverter
import TableExtractor as TE
from BoxExtractor import ImageProcessor
import os
import os
import pytesseract
from PIL import Image
import pandas as pd
from natsort import natsorted

pdf_file = 'data/SAMPLETEXT.pdf'
output_folder = 'images'

# Create an instance of the PDFToImagesConverter class
pdf_converter = PDFToImagesConverter(pdf_file, output_folder)

# Call the convert_to_images method to convert and save the images
pdf_converter.convert_to_images()


path_to_image = "./images/page_1.jpg"

# Create an instance of the Table Extractor class
table_extractor = TE.TableExtractor(path_to_image)

# Call the execute method in Table extractor to get the image of the table in pdf
table_image = table_extractor.execute()

#save the results from table extractor
cv2.imwrite("perspective_corrected_image.jpg", table_image)

#Create an instance of Image Processor 
processor = ImageProcessor()

#Call box_extraction to individually extract boxes in the table
processor.box_extraction("perspective_corrected_image.jpg")


folder_path = "cropped_boxes"

# Get a list of image files in the folder and sort them naturally
image_files = natsorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
# Initialize empty lists to store OCR results
even_text = []
odd_text = []

# Loop through the sorted image files and perform OCR
for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    text.strip()

    # Determine if the image is even or odd
    if i % 2 == 0:
    
        even_text.append(text)
        
    else:
        odd_text.append(text)


# Create a DataFrame

df= pd.DataFrame({"First Row": odd_text, "2nd Row": even_text})


# Replace line feed and control character with a space in the entire DataFrame
df = df.apply(lambda x: x.str.replace('&#10;_x000c_', ' '))
df = df.apply(lambda x: x.str.replace('_x000C_', ' '))


# # To save the DataFrame to a CSV file and excel file
df.to_csv("ocr_results.csv", index=False)
df.to_excel("ocr_table.xlsx", index = False)




