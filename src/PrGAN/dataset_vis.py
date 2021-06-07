#! usr/bin/env python3

from numpy import load
from matplotlib import pyplot as plt

def plot_faces(faces, n):
    for i in range(n*n):
        plt.subplot(n,n, 1+i)
        plt.axis('off')
        plt.imshow(faces[i].astype('int8'))
    plt.show()

if __name__ == '__main__':
    dir = ''
    data = load(dir)
    faces = data['arr_0']
    print('Loaded : ', faces.shape)
    plot_faces(faces, 10)