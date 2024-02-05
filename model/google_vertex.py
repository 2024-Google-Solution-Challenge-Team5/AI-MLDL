import os
import io
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
import cv2
from google.cloud import vision
import matplotlib.pyplot as plt

def plt_imshow(title="image", img=None, figsize=(8, 5)):
    """
    Display image using matplotlib.
    Args:
        title (str, optional): Image title  Defaults to "image".
        img (_type_, optional): Real Image Defaults to None.
        figsize (tuple, optional): Plot Size. Defaults to (8, 5).
    """
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
        
def putText(image, text, x, y, color=(0, 255, 0), font_size=22):
    """
    Image의 Bounding Box에 Text를 넣어주는 함수
    Args:
        image (_type_): 
        text (_type_): 
        x (_type_): 
        y (_type_): 
        color (tuple, optional): . Defaults to (0, 255, 0).
        font_size (int, optional): . Defaults to 22.

    Returns:
        _type_: 
    """
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    if platform.system() == "Darwin":
        font = "AppleGothic.ttf"
    elif platform.system() == "Windows":
        font = "malgun.ttf"
    else:
        font = "NanumGothic.ttf"

    image_font = ImageFont.truetype(font, font_size)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

        
def is_disposal_box(img_path,response,texts):
    img = cv2.imread(img_path)
    roi_img = img.copy()
    disposal_box_flag = False
    for text in texts:
        if "수거" not in text.description or len(text.description) > 10:
            disposal_box_flag = False
            continue
        else:
            disposal_box_flag = True
            print('\n"{}"'.format(text.description))

            vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

            ocr_text = text.description
            x1 = text.bounding_poly.vertices[0].x
            y1 = text.bounding_poly.vertices[0].y
            x2 = text.bounding_poly.vertices[1].x
            y2 = text.bounding_poly.vertices[2].y

            cv2.rectangle(roi_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            roi_img = putText(roi_img, ocr_text, x1, y1 - 30, font_size=30)
            
            break

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    
    if disposal_box_flag:
        return roi_img,True
    else:
        print("수거함이 아닙니다!!!")
        return img,False

def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jun/Development/AI-MLDL/drugbox-service-account.json"

    client = vision.ImageAnnotatorClient()

    img_path = "/Users/jun/Development/AI-MLDL/images/disposal_box.jpeg"
    with io.open(img_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    roi_img, is_disposal = is_disposal_box(img_path, response, texts)
    if is_disposal:
        print("It is right Disposal Box")
        plt_imshow("ROI", roi_img, figsize=(16, 10))
    

if __name__ == "__main__":
    main()