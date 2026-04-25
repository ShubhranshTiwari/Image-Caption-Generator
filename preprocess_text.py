import os
import string
import pickle

def load_doc(filename):
    """Loads the captions file into memory."""
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_descriptions(doc):
    """Maps each image ID to its list of 5 captions."""
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        # Remove extension from image id (e.g., .jpg)
        image_id = image_id.split('.')[0]
        # Join description tokens back (in case of commas in text)
        image_desc = " ".join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping

def clean_descriptions(descriptions):
    """Cleans the captions: lowercase, remove punctuation, remove small words."""
    # Prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # Tokenize
            desc = desc.split()
            # Convert to lower case
            desc = [word.lower() for word in desc]
            # Remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # Remove hanging 's' and 'a' (1-character words)
            desc = [word for word in desc if len(word) > 1]
            # Remove tokens with numbers
            desc = [word for word in desc if word.isalpha()]
            # Wrap with start and end tokens
            desc_list[i] =  'startseq ' + ' '.join(desc) + ' endseq'

def create_vocabulary(descriptions):
    """Extracts a set of all unique words from the cleaned captions."""
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# --- Execution ---
filename = 'captions.txt' # Path to your Flickr8k captions file
doc = load_doc(filename)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)
vocabulary = create_vocabulary(descriptions)

print(f"Loaded: {len(descriptions)} images")
print(f"Vocabulary Size: {len(vocabulary)}")

# Save the cleaned descriptions to a file for the next step
with open('descriptions.pkl', 'wb') as f:
    pickle.dump(descriptions, f)