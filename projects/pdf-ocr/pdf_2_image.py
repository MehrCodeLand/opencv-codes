import fitz 
from PIL import Image
import io 


def pdf_to_jpg(pdf_path , output_folder):
    pdf_document = fitz.open(pdf_path)

    counter = 0 
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()

        image = Image.open(io.BytesIO(pix.tobytes("png")))
        image_path = f"{output_folder}/page_{page_num + 1}.jpeg"
        image.save(image_path,"JPEG")
        counter = counter + 1

        if counter == 5 :
            break




pdf_path = 'pd.pdf'
output_folder = 'images'
pdf_to_jpg(pdf_path,output_folder)