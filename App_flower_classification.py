
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore") 
import streamlit as st
import time
import wikipedia as wiki
from rake_nltk import Rake
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
    prob=np.max(prediction)
    return pred,prob

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

st.write(f"There are a total of {len(labels)} classes in the training dataset") 
st.success("Loaded the Model :)")



file = st.file_uploader("Choose an image...", type="jpeg")
if file is not None:
    with st.spinner('Wait for it...'):
        
            image = Image.open(file)
            st.image(image, caption='Uploaded Image.', width=None)
            
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)
            prediction,prob=normalize_image(image_array)           
            st.write("The flower is more likely to be a/an ",labels[prediction].upper(),"with a probability of ",prob )
            
            command=f"{labels[prediction]} flower"
            rake = Rake()
            rake.extract_keywords_from_text(command)
            key = rake.get_ranked_phrases()
            rake.get_ranked_phrases_with_scores()
            st.write("Key: " + str(key))
            st.subheader("Here is some information about the predicted flower:)")
            try:
                page = wiki.page(key)
                st.info(page.summary)
            except:
                topics = wiki.search(key)
                st.write(f"{labels[prediction].upper()} may refer to: ")
                for i, topic in enumerate(topics):
                    st.write(i, topic)
                choice = st.text_input("Enter a choice: ")
                assert int(choice) in xrange(len(topics))
                st.info(wiki.summary(topics[choice])
                
   st.subheader("Thank You :)")                    
else:
    st.warning("No file has been chosen yet")


