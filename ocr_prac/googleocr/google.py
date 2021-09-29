import os, io
import json


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/lab12/recipebranch/RECIPENLGforMC/ocr_prac/googleocr/multicampusproject-69b320477a68.json"
def detect_text(path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))
        with open("./ocrout") as f:
            

file_name = os.path.join(
    os.path.dirname(__file__),
    '/home/lab12/recipebranch/RECIPENLGforMC/ocr_prac/test_img.jpg')

detect_text(file_name)