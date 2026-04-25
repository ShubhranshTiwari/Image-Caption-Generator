import streamlit as st
import numpy as np
import pickle
import time
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Image Captioner", page_icon="🖼️", layout="wide")

# --- CACHE MODELS ---
@st.cache_resource
def load_all_models():
    # Load your trained model
    model = load_model('model_14.keras') # This is now your smartest model
    # Load feature extractor
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    # Load tokenizer
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    return model, vgg_model, tokenizer

model, vgg_model, tokenizer = load_all_models()
max_length = 34

# --- HELPER FUNCTIONS ---
def extract_features(img_path, model):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# def generate_caption(model, tokenizer, photo, max_length):
#     in_text = 'startseq'
#     top_5_data = None
    
#     for i in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
        
#         # Predict probabilities for all words in vocabulary
#         preds = model.predict([photo, sequence], verbose=0)[0]
        
#         # Capture top 5 for the FIRST word only
#         if i == 0:
#             top_indices = preds.argsort()[-5:][::-1]
#             top_5_data = {tokenizer.index_word[idx]: preds[idx] for idx in top_indices if idx in tokenizer.index_word}
        
#         yhat = np.argmax(preds)
#         word = tokenizer.index_word.get(yhat)
        
#         if word is None or word == 'endseq':
#             break
            
#         in_text += ' ' + word
        
#     return in_text.replace('startseq', '').strip(), top_5_data

def beam_search_predictions(model, tokenizer, image_features, max_length, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    beams = [[start, 0.0]]
    top_5_first_word = {} # To store the probs for the bar chart

    for i in range(max_length):
        all_candidates = list()
        for seq, score in beams:
            if seq[-1] == tokenizer.word_index.get('endseq'):
                all_candidates.append([seq, score])
                continue
                
            sequence = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([image_features, sequence], verbose=0)[0]
            
            # Capture top 5 probabilities for the bar chart (only on the first iteration)
            if i == 0 and not top_5_first_word:
                top_indices = np.argsort(preds)[-5:][::-1]
                top_5_first_word = {tokenizer.index_word[idx]: float(preds[idx]) for idx in top_indices if idx in tokenizer.index_word}

            top_k = np.argsort(preds)[-beam_index:]
            for word_idx in top_k:
                candidate_seq = seq + [word_idx]
                candidate_score = score + np.log(preds[word_idx] + 1e-10)
                all_candidates.append([candidate_seq, candidate_score])
        
        beams = sorted(all_candidates, key=lambda l: l[1], reverse=True)[:beam_index]
        if all(seq[-1] == tokenizer.word_index.get('endseq') for seq, score in beams):
            break

    best_seq = beams[0][0]
    final_caption = [tokenizer.index_word[i] for i in best_seq if i not in [tokenizer.word_index['startseq'], tokenizer.word_index.get('endseq')]]
    
    # Returning BOTH the string and the dictionary to match your app's expectations
    return ' '.join(final_caption), top_5_first_word

# --- SIDEBAR ---
st.sidebar.title("🤖 Model Details")
st.sidebar.info("""
**Architecture:** CNN (VGG16) + LSTM  
**Dataset:** Flickr8k  
**Epochs Trained:** 15  
""")
st.sidebar.markdown("---")
st.sidebar.write("Created by: Shubhransh Tiwari")

# --- MAIN UI ---
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image, and our deep learning model will describe what it 'sees'!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        # Save temp file for VGG16 to read
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

            with st.expander("📊 Model Performance Metrics"):
                st.write("Current Model: `model_9.keras`")
                st.metric(label="BLEU-1 Score", value="0.50", delta="0.05")
                st.metric(label="BLEU-4 Score", value="0.15", delta="0.02")
                st.caption("Scores based on Flickr8k test set evaluation.")

    with col2:
        st.subheader("📝 Generated Caption")
        
        with st.spinner('Analyzing image features...'):
            photo_feature = extract_features("temp_image.jpg", vgg_model)
            # Unpack both the caption and the top 5 probabilities
            # caption, top_5_data = generate_caption(model, tokenizer, photo_feature, max_length)
            caption, top_5_data = beam_search_predictions(model, tokenizer, photo_feature, max_length, beam_index=3)

        st.markdown("**Caption:**")# Clean and show the caption text
        final_caption = caption.replace('endseq', '').strip()
        st.subheader(f"Generated Caption: {final_caption}")

        # Show the Bar Chart for word probabilities
        if top_5_data:
            st.write("### 📊 Word Confidence (First Word)")
            # We use a bar chart to show the top 5 candidates the model considered
            st.bar_chart(top_5_data)
    
            # Optional: Display as a nice table for specific percentages
            import pandas as pd
            df = pd.DataFrame(top_5_data.items(), columns=['Word', 'Probability'])
            st.dataframe(df.set_index('Word'), use_container_width=True)    

        # Typewriter effect for a "cool" display
        placeholder = st.empty()
        full_caption = ""
        for word in caption.split():
            full_caption += word + " "
            placeholder.markdown(f"### *{full_caption}*")
            time.sleep(0.1)
        
        st.success("Analysis Complete!")
        
        # Additional project details
        with st.expander("Show Technical Breakdown"):
            st.write(f"**Image Vector Shape:** {photo_feature.shape}")
            st.write(f"**Vocabulary Match:** Found {len(caption.split())} words")
            st.write("**Algorithm:** Beam Search with Beam Width = 3")

else:
    st.info("Please upload a JPG or PNG image to start.")