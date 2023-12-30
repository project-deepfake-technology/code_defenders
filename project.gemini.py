!pip install tensorflow tensorflow-gpu mathplotlib tensorflow-datasets ipywidgets
!pip list 
import tensorflow as tf:
gpus=tf.config.expermintal.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, TRUE)
import tensorflow_datasets as tfds
from mathplotlib import pyplot as plt
ds=tfds.load('fashion_mnist',split='train')
ds.as_numpy_iterator().next()['label']
   ------------- STEP2:--------------------------
import numpy as np
dataiterator =ds.as_numpy_iterator()
dataiterator.next()
#image visualization code
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx in range(4):
    sample=dataiterator.next()
    ax[idx].imshow(np.squeeze(batch['image']))
    ax[idx].title.set_text(batch['label'])
#image visualization ending
#pic are in 0 to 255 and convert to 0 to 1
  def scale_images(data):
      image=data['image']
      return image / 255
ds=ds.tfds.load('fashion_mnist',split='train')
ds=ds.map(scale_images)
ds=ds.cache() 
ds=ds.shuffle(60000)
ds=ds.batch(128)
ds=ds.prefetch(64)
ds.as_numpy_iteration().next().shape
#(128,28,28,7)length,breath,height
 ------------- STEP3:--------------------------
#building image generator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Reshape,LeakyReLU,LeakyReLU,Dropout,UpSampling2D
def build_generator():
    model.add(Dense(7*7*128,input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
#upsampling block1
    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))
#upsampling block2
    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))
#down sampling block1
    model.add(Conv2D(128,4,padding='same'))
    model.add(LeakyReLU(0.2))
#Downsampling block2
    model.add(Conv2D(128,4,padding='same'))
    model.add(LeakyReLU(0.2))
#conv many layer to one channel
model.add(Conv2D(1,4,padding='same',activation='sigmond'))
return model
generator = build_generator()
generator.summary()
img=generator.predict(np.random.randn(4,128,1))
img.shape
#example i.e testing the generator
img=generator.predict(np.random.randn(4,128,1))
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx, img  in enumerate(img):
    sample=dataiterator.next()
    ax[idx].imshow(np.squeeze(img))
    ax[idx].title.set_text(idx)

----------------------------------step4------------------
# constructing discrimantor
def build_discriminator():
model=Sequential()
#first conv block
  model.add(Conv2D(32,5,input_shape=(28,28,1)))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4)
#second conv block 
  model.add(Conv2D(64,5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4)
#third conv block 
  model.add(Conv2D(128,5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4)
#fourth conv block 
  model.add(Conv2D(256,5))
  model.add(LeakyReLU(0.2))
  model.add(Dropout(0.4)
#flattern the the image
model.add(flattern())
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))#1 represent the flase image and 0 represent the true image
return model
discriminator=build_discriminator()
discriminator.summary()
img=img[0]
img.shape
discriminator.predict(img)
----------------------------------step5--------------------------------
#creating custom loops b/w generator and discriminator
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import BinaryCrossentropy
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()
#importing the base model
from tensorflow.keras.model import Model
tf.random.normal((6,28,28,1))
class FashionGAN(Model):
    def__init__(self,generator,discriminator,*args,**kwargs):
       super().__init__(*args,**kwargs)
       self.generator=generator 
       self.discriminator=discriminator
    def compile(self,g_opt,d_opt,g_loss,d_loss,*args,**kwargs):
       super().compile(*args,**kwargs)
       self.g_opt=g_opt
       self.d_opt=d_opt
       self.g_loss=g_loss
       self.d_loss=d_loss
   def train_step(self):
       real_images = batch
       fake_images = self.generator(tf.random.normal((128,128,1)),training=False)
#train discriminator
with tf.GradientTape() as d_tape:
yhat_real = self.discriminator(real_images,training=True)
yhat_fake = self.discriminator(fake_images,training=True)
yhat_realfake = tf.concat([yhat_real,yhat_fake],axis=0)
#create lbl for real,fake img
y_realfake = tf.concat([tf.zeros_like(yhat_real),tf.ones_like(yhat_fake)],axis=0)
#output noise
noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
noise_fake = 0.15*tf.random.uniform(tf.shape(yhat_fake))
y_realfake +=tf.concat([noise_real,noise_fake],axis=0)
#calculate loss
total_d_loss = self.d_loss(y_realfake,yhat_realfake)
#nn learn
dgradn = d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
self.d_opt.apply_gradients(zip(dgrad,self.discriminator.trainable_variables))
with tf.GradientTape() as g_tape:
    #gen new img
     gen_images = self.generator(tf.random.normal((128,128,1)),training=True)
     #predict labels
     predicted_labels = self.discriminator(gen_images, training=False)
     #cal loss
     total_g_loss = self.g_loss(tf.zeros_like(predicted_labels),predicted_labels)
     #apply bkprop
     ggrad = g_tape.gradient(total_g_loss,self.generator.trainable_variables)
     self.g_opt.apply_gradients(zip(ggrad,self.generator.trainable_variables))
    
     return{"d_loss":total_d_loss,"g_loss":total_g_loss}
     

#create sub cls
    fashgan = FashionGAN(generator,discriminator)
    fashgan.compile(g_opt,d_opt,g_loss,d_loss)
    
import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.Callbacks import Callbacks

class ModelMonitor(Callbacks):
    def__init__(self,num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
    def on_epoch_end(self,epoch,logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img,self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images',f'generated_images_{epoch}_{i}.png'))
#training
hist = fashgan.fit(ds,epoch=20,callbacks=[ModelMonitor()])
---------------------------finally-----------------------------
#generate img
generator.load_weights(os.path.join('archive','generatormodel.h5'))
imgs = generator.predict(tf.random.normal((16,128,1)))
fig,ax = plt.subplots(ncols=4,nrows=4,figsize=(20,20))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(imgs[r+1]*[c+1]-1)

-----------saving img ---------------------
