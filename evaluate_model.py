import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --- LOAD DATA ---
descriptions = pickle.load(open('descriptions.pkl', 'rb')) # Your cleaned descriptions
test_features = pickle.load(open('features.pkl', 'rb'))    # Your image features
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = load_model('model_9.keras')
max_length = 34

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

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    count = 0
    total = len(descriptions)
    # Iterate over the entire test set (or a subset to save time)
    for i, (key, desc_list) in enumerate(descriptions.items()):
        if i > 100: break  # Just do the first 100 images
        if key not in photos: continue

        count += 1
        if count % 10 == 0: # Print every 10 images
            print(f"Processing image {count}/{total}...")
            
        # Reshape the feature to (1, 4096) to add the batch dimension
        image_feature = photos[key][0].reshape((1, 4096))

        # Use the reshaped feature in the function call
        yhat_text, _ = beam_search_predictions(model, tokenizer, image_feature, max_length, beam_index=3)
        
        # Now yhat_text is a string, and .split() will work
        yhat_clean = yhat_text.split()
        
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat_clean)
        
    # Calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# Run the evaluation
evaluate_model(model, descriptions, test_features, tokenizer, max_length)