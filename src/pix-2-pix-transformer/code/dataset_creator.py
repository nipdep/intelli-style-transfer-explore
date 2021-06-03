
import os
from numpy import asarray, vstack, savez_compressed
from keras.preprocessing.image import img_to_array, load_img

def preprocess_img_file(path :str, size=(256,512)):
    src_list, target_list = [], []
    for file in os.listdir(path):
        pixels = load_img(os.path.join(path,file), target_size=size)
        pixels = img_to_array(pixels)
        src_img, target_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(src_img)
        target_list.append(target_img)
    return (asarray(src_list),  asarray(target_list))

if __name__ == '__main__':

    path = './data/maps/train'
    src_imgs, target_imgs = preprocess_img_file(path)
    print(f"src image dataset shape : {src_imgs.shape}")
    print(f"target img dataset shape : {target_imgs.shape}")
    filename = "maps_256.npz"
    savez_compressed(os.path.join(path,filename), src_imgs, target_imgs)
    

    
