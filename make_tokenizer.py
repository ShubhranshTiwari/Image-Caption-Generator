import pickle
from keras.src.legacy.preprocessing.text import Tokenizer

# 1. Load your cleaned descriptions
descriptions = pickle.load(open('descriptions.pkl', 'rb'))

# 2. Extract all captions into a list
all_captions = []
for key in descriptions:
    for cap in descriptions[key]:
        all_captions.append(cap)

# 3. Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

# 4. Save it!
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("tokenizer.pkl has been created successfully!")