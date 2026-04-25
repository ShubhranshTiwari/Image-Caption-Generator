import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

def extract_features(directory):
    # Load the VGG16 model
    model = VGG16()
    # Re-structure the model: remove the last layer (prediction layer)
    # We want the output from the 'fc2' layer (4096 features)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = dict()
    
    # Loop through every image in the folder
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        
        # 1. Load and resize image to 224x224
        image = load_img(img_path, target_size=(224, 224))
        # 2. Convert pixel values to a numpy array
        image = img_to_array(image)
        # 3. Reshape data for the model (add a batch dimension)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # 4. Preprocess for VGG16 (standard normalization)
        image = preprocess_input(image)
        
        # 5. Extract features
        feature = model.predict(image, verbose=0)
        
        # 6. Store feature in dictionary with image ID as key
        image_id = img_name.split('.')[0]
        features[image_id] = feature
        
    return features

# --- Execution ---
# Set the path to your Flickr8k images folder
directory = 'data/Flickr8k_Images' 
features = extract_features(directory)

# Save the features to a file (so you never have to run this heavy task again)
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print(f"Extracted Features: {len(features)}")