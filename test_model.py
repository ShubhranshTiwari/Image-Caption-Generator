import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

# 1. Load the required files
# Use the model from the last epoch (e.g., model_19.keras)
model = load_model('model_9.keras')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
max_length = 34  # This must match the max_length from training

# 2. Setup the feature extractor (VGG16) again for the test image
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_single_image_features(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# def generate_caption(model, tokenizer, photo, max_length):
#     # Start the sequence
#     in_text = 'startseq'
#     # Iterate over the max length of sequences
#     for i in range(max_length):
#         # Integer encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # Pad input
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         # Predict next word
#         yhat = model.predict([photo, sequence], verbose=0)
#         # Convert probability to integer
#         yhat = np.argmax(yhat)
#         # Map integer to word
#         word = word_for_id(yhat, tokenizer)
#         # Stop if we cannot map the word
#         if word is None:
#             break
#         # Append as input for generating the next word
#         in_text += ' ' + word
#         # Stop if we predict the end of the sequence
#         if word == 'endseq':
#             break
#     return in_text

def generate_caption_with_probs(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    top_5_data = None
    
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict probabilities for all words in vocabulary
        preds = model.predict([photo, sequence], verbose=0)[0]
        
        # Capture top 5 for the FIRST word only
        if i == 0:
            top_indices = preds.argsort()[-5:][::-1]
            top_5_data = {tokenizer.index_word[idx]: preds[idx] for idx in top_indices if idx in tokenizer.index_word}
        
        yhat = np.argmax(preds)
        word = tokenizer.index_word.get(yhat)
        
        if word is None or word == 'endseq':
            break
            
        in_text += ' ' + word
        
    return in_text.replace('startseq', '').strip(), top_5_data

# --- TEST ON A NEW IMAGE ---
image_path = 'data/Flickr8k_Images/1000268201_693b08cb0e.jpg' # Put any image path here!
photo_feature = extract_single_image_features(image_path)
# The function now returns (string, dictionary)
caption_text, top_5_data = generate_caption_with_probs(model, tokenizer, photo_feature, max_length)

# Clean the text variable we just unpacked
final_caption = caption_text.replace('startseq', '').replace('endseq', '').strip()
print(f"Generated Caption: {final_caption}")