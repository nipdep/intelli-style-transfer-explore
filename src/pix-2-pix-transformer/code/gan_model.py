#!usr/bin/env python3
#%%
from keras.initializers import RandomNormal
from keras.models import Input, Model
from keras.layers import Concatenate, Conv2D, BatchNormalization, LeakyReLU, Activation
from keras.optimizers import Adam
#%%

def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    #input layer
    src_img = Input(shape=image_shape)
    tar_img = Input(shape=image_shape)
    merged = Concatenate([src_img, tar_img])
    #c64
    d = Conv2D(64, (4, 4), (2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    #c128
    d = Conv2D(128, (4, 4), (2, 2), padding='same', kernel_initializer=init)(merged)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d) 
    #c256
    d = Conv2D(256, (4, 4), (2, 2), padding='same', kernel_initializer=init)(merged)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #c512
    d = Conv2D(512, (4, 4), (2, 2), padding='same', kernel_initializer=init)(merged)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(64, (4, 4), padding='same', kernel_initializer=init)(merged)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([src_img, tar_img], patch_out)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model 
#%%
model = define_discriminator((128,128,3))
model.summary()

# %%
