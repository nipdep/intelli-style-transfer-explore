#!usr/bin/env python3
#%%
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.models import Input, Model
from keras.layers import Concatenate, Conv2D, BatchNormalization, LeakyReLU, Activation, Conv2DTranspose, Dropout
from keras.optimizers import Adam
#%%

def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    #input layer
    src_img = Input(shape=image_shape)
    tar_img = Input(shape=image_shape)
    merged = Concatenate()([src_img, tar_img])
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

model = define_discriminator((128, 128, 3,))
model.summary()

tf.keras.utils.plot_model(model=model, show_shapes=True, dpi=76)

def define_encoder_block(layer_in, n_filers, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filers, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init =RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g

def define_generator(image_shape=(256,256,3)):
    init = RandomNormal(stddev=0.02)

    in_image = Input(shape=image_shape)
    #encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    #bottleneck layer [latent vector]
    b = Conv2D(512, (4, 4), (2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    #decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    g = Conv2DTranspose(3, (4, 4), (2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('relu')(g)
    model = Model(in_image, out_image)
    return model 

gen = define_generator()
gen.summary()

tf.keras.utils.plot_model(model=gen, show_shapes=True, dpi=76)
# %%

def define_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    gan_model = Model(in_src, [dis_out, gen_out])
    opt = Adam(lr=0.002, beta_1=0.5)
    gan_model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
