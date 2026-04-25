import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

# 1. Load your saved data
descriptions = pickle.load(open('descriptions.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

# 2. Prepare Tokenizer
all_captions = []
for key in descriptions:
    for cap in descriptions[key]:
        all_captions.append(cap)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(d.split()) for d in all_captions)

# 3. Data Generator (Prevents Memory Crash)
def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            # CHECK: Skip the key if it's not in our features (like the 'image' header)
            if key not in features:
                continue
                
            n += 1
            photo = features[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            
            if n == batch_size:
                yield ((np.array(X1), np.array(X2)), np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0

# 4. Define the Model Architecture
# Image Feature Input (Encoder)
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Sequence Input (Decoder)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Merge both models
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

import tensorflow as tf

# 5. Updated Training Logic for TF 2.16+ / Python 3.13
import os
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
total_target_epochs = 15
last_finished_epoch = 10  # You already have model_9.keras
batch_size = 32
steps = len(descriptions) // batch_size

# 1. LOAD PREVIOUS MODEL
# We load 'model_9.keras' (which is the result of the 10th epoch)
model_path = 'model_9.keras'

if os.path.exists(model_path):
    print(f"Resuming training from {model_path}...")
    model = load_model(model_path)
else:
    print("Previous model not found. Starting from scratch.")
    last_finished_epoch = 0

# 2. UPDATED TRAINING LOOP
# initial_epoch=10 and epochs=15 means it will run epochs 11, 12, 13, 14, and 15.
output_signature = (
    (
        tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),   # Image features
        tf.TensorSpec(shape=(None, max_length), dtype=tf.int32) # Text sequences
    ),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)  # Target word
)
for i in range(last_finished_epoch, total_target_epochs):
    # Create the generator instance
    gen = data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size)
    
    # Wrap in dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: gen,
        output_signature=output_signature
    )
    
    print(f"Epoch {i+1}/{total_target_epochs}")
    
    # Pass initial_epoch to fit so it knows it's not starting at 0
    model.fit(dataset, steps_per_epoch=steps, verbose=1)
    
    # Save with the correct index (model_10.keras, model_11.keras, etc.)
    model.save(f'model_{i}.keras')