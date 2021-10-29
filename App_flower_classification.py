
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore") 
import glob
import cv2
import os
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator

st.title('Image Classification - Flower Dataset Project')
# Load the model
st.subheader("Loading the model...")
model = load_model('keras_model.h5')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
         
def normalize_image(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #resize the image to a 224x224 
    #resizing the image to be at least 224x224 and then cropping from the center
    #size = (224, 224)  
    # Normalize the image
    normalized_image_array = (img.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    #print(prediction)
    pred=np.argmax(prediction)
    return pred

labels=['alpine sea holly',
 'anthurium',
 'artichoke',
 'azalea',
 'balloon flower',
 'barberton daisy',
 'bee balm',
 'bird of paradise',
 'bishop of llandaff',
 'black-eyed susan',
 'blackberry lily',
 'blanket flower',
 'bolero deep blue',
 'bougainvillea',
 'bromelia',
 'buttercup',
 'californian poppy',
 'camellia',
 'canna lily',
 'canterbury bells',
 'cape flower',
 'carnation',
 'cautleya spicata',
 'clematis',
 "colt's foot",
 'columbine',
 'common dandelion',
 'common tulip',
 'corn poppy',
 'cosmos',
 'cyclamen_',
 'daffodil',
 'daisy',
 'desert-rose',
 'fire lily',
 'foxglove',
 'frangipani',
 'fritillary',
 'garden phlox',
 'gaura',
 'gazania',
 'geranium',
 'giant white arum lily',
 'globe thistle',
 'globe-flower',
 'grape hyacinth',
 'great masterwort',
 'hard-leaved pocket orchid',
 'hibiscus',
 'hippeastrum_',
 'iris',
 'japanese anemone',
 'king protea',
 'lenten rose',
 'lilac hibiscus',
 'lotus',
 'love in the mist',
 'magnolia',
 'mallow',
 'marigold',
 'mexican petunia',
 'monkshood',
 'moon orchid',
 'morning glory',
 'orange dahlia',
 'osteospermum',
 'passion flower',
 'peruvian lily',
 'petunia',
 'pincushion flower',
 'pink primrose',
 'pink quill',
 'pink-yellow dahlia',
 'poinsettia',
 'primula',
 'prince of wales feathers',
 'purple coneflower',
 'red ginger',
 'rose',
 'ruby-lipped cattleya',
 'siam tulip',
 'silverbush',
 'snapdragon',
 'spear thistle',
 'spring crocus',
 'stemless gentian',
 'sunflower',
 'sweet pea',
 'sweet william',
 'sword lily',
 'thorn apple',
 'tiger lily',
 'toad lily',
 'tree mallow',
 'tree poppy',
 'trumpet creeper',
 'wallflower',
 'water lily',
 'watercress',
 'wild geranium',
 'wild pansy',
 'wild rose',
 'windflower',
 'yellow iris']
st.success("Loaded the Model :)")
option = st.multiselect('Would you like to choose random image from test dataset or upload image?',["Select Random Image from Test Dataset", "Upload from test dataset"])
if "Select Random Image from Test Dataset" in option:

    all_images = [cv2.imread(file) for file in glob.glob("test//*.jpeg")]
    st.write(f"There are totally {len(all_images)} images in the test dataset")
    img_chosen=st.slider('Choose a number:', 0, len(all_images))
    st.write("By default, it chooses the first image in the test dataset")
    a_img=all_images[img_chosen]
    st.image(a_img, caption='Choosen Image', width=None)
    st.subheader("Predicting...")
    prediction=normalize_image(a_img)
    st.write("The flower is more likely to be a/an ", labels[prediction].upper())
    st.success("Thank You :)")

elif "Upload from test dataset" in option:
    file = st.file_uploader("Choose an image...", type="jpeg")
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image.', width=None)
        
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        prediction=normalize_image(image_array)           
        st.write("The flower is more likely to be a/an",labels[prediction].upper())
        st.subheader("Thank You :)")
    else:
         st.warning("No file has been chosen yet")


