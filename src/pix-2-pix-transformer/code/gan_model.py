#!usr/bin/env python3
#%%
from numpy.core.function_base import add_newdoc
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
    d = Conv2D(128, (4, 4), (2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d) 
    #c256
    d = Conv2D(256, (4, 4), (2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    #c512
    d = Conv2D(512, (4, 4), (2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(64, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([src_img, tar_img], patch_out)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model 

model = define_discriminator((256,256,3,))
model.summary()

#tf.keras.utils.plot_model(model=model, show_shapes=True, dpi=76)

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

#tf.keras.utils.plot_model(model=gen, show_shapes=True, dpi=76)
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
    return gan_model
#%%
from numpy.random import randint
from numpy import ones, zeros, load

def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def generate_real_examples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 =  trainA[ix], trainB[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def generate_fake_examples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y   

#%%
from matplotlib import pyplot
def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_realA,  X_realB], _ = generate_real_examples(dataset, 3, 1)
    X_fakeB, _ = generate_fake_examples(g_model, X_realA, 1)
    X_realA = (X_realA+1)/2.0
    X_realB = (X_realB+1)/2.0
    X_fakeB = (X_fakeB+1)/2.0

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1+ i)
        pyplot.axis("off")
        pyplot.imshow(X_realA[i])

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(X_fakeB[i])

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])

    
    filename = 'plot_%06d.png'%(step+1)
    pyplot.savefig(filename)
    pyplot.close()

    modelfile = 'gen_model_%06d.h5'%(step+1)
    g_model.save(modelfile)
    print('>Saved: %s and %s'%(filename,  modelfile))

#%%

def train(d_model, g_model, gan_model, dataset, n_epochs, n_batch=1):
    n_patchs = d_model.output_shape[1]
    trainA, trainB= dataset
    bat_per_epoch = int(len(trainA)/n_batch)
    n_steps = bat_per_epoch*n_epochs
    for i in range(n_steps):
        [X_realA, X_realB],y_real = generate_real_examples(dataset, n_batch, n_patchs)
        X_fake, y_fake = generate_fake_examples(g_model, X_realA, n_patchs)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB],y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fake],y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        if ((i+1)%10)==9:
            summarize_performance(i, g_model, dataset)

# %%

dataset = load_real_samples('../../../data/data/maps/maps_256.npz')
print('Loaded : ', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
train(d_model, g_model, gan_model, dataset, n_epochs=3, n_batch=16)
# %%


# %%
