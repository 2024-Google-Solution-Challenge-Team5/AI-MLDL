import easyocr

reader = easyocr.Reader(["ko", "en"])  # this needs to run only once to load the model into memory
result = reader.readtext("images/처방약_예시.jpeg", detail=0)

print(result)
