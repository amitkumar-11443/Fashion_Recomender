import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm   #linalg = Linear algebra
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))  # Model scale down image 224,224,3 
                                                                                # incluse False as we creating top layer
                                                                                # We are not training model we are using
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()                                                        # this is our top layer
])

#print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

#print(len(filenames)) # check the file length
#print(filenames[0:5])  # check file names for the fist 5 file

feature_list = []

for file in tqdm(filenames):        # tqdm gives the for loop progress , instead of "for file in filenames"
    feature_list.append(extract_features(file,model))

#print(np.array(feature_list).shape)


pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))


