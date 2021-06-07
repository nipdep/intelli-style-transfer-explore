#! usr/bin/env python3

from os import listdir
from numpy import asarray, savez_compressed
from PIL import Image
from mtcnn.mtcnn import MTCNN         
from matplotlib import pyplot as plt 

def load_image(filename):
    image = Image(filename)
    image = image.convert("RGB")
    pixel = asarray(image)
    return pixel

def extract_face(model, pixels, required_size=(128, 128)):
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1+width, y1+height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_faces(dir, n_faces):
    model = MTCNN()
    faces = []
    for filename in listdir(dir):
        pixels = load_image(filename)
        face = extract_face(model, pixels)
        if face is None:
            continue
        faces.append(face)
        print(len(faces), face.shape)
        if len(faces) > n_faces:
            break
        return asarray(faces)

if __name__ == '__main__':
    directory = 'img_align_celeba/'
    all_faces = load_faces(directory, 50000)
    print('Loaded : ', all_faces.shape)
    savez_compressed('img_align_celeba_128.npz')
