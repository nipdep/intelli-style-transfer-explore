
from numpy import load
from matplotlib import pyplot as plt

data = load('./data/maps/maps_256.npz')
src_images, tar_images =data['arr_0'], data['arr_1']
print('loaded : ', src_images.shape, tar_images.shape)
n_samples =3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1+i)
    plt.axis('off')
    plt.imshow(src_images[i].astype('uint8'))
    plt.subplot(2, n_samples, 1+i+n_samples)
    plt.imshow(tar_images[i].astype('uint8'))
plt.show()

print("FINISH")