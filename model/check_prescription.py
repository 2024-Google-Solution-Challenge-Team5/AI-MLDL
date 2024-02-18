import os
import io
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
import cv2
from google.cloud import vision
import matplotlib.pyplot as plt
import re

path = "../images/script.jpeg"
client = vision.ImageAnnotatorClient()

with io.open(path, "rb") as image_file:
    content = image_file.read()
image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations
img = cv2.imread(path)
roi_img = img.copy()

for text in texts:
    if  not (re.search("(정|캡슐)$", text.description)) or len(text.description) > 10 :
        continue
    print('\n"{}"'.format(text.description))

    vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

    ocr_text = text.description
    x1 = text.bounding_poly.vertices[0].x
    y1 = text.bounding_poly.vertices[0].y
    x2 = text.bounding_poly.vertices[1].x
    y2 = text.bounding_poly.vertices[2].y

    cv2.rectangle(roi_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    roi_img = putText(roi_img, ocr_text, x1, y1 - 30, font_size=10)

if response.error.message:
    raise Exception(
        "{}\nFor more info on error messages, check: "
        "https://cloud.google.com/apis/design/errors".format(response.error.message)
    )

plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))