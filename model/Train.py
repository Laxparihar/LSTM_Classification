import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
from sklearn.utils import shuffle
import pickle
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the dataset
df = pd.read_csv("./data.csv")
print(df.head())

# Shuffle the dataset and reset the index
df = shuffle(df).reset_index(drop=True)

# Get class distribution
class_legnth = df['Class'].value_counts()
print('Check Null Values:', df.isnull().sum(), '\n---')
print('Check unique Classes:', df['Class'].unique(), '\n---')
print('Check Datatypes:', df.dtypes, '\n---')

# Keep only 'Text' and 'Class' columns
df = df[['Text', 'Class']]

# Convert 'Text' column to string data type
df['Text'] = df['Text'].astype(str)

# Load stopwords
stop = stopwords.words('english')

# Define a function to clean the text
def clean_text(msg):
    # Convert text to lowercase
    msg = msg.lower()
    # Tokenize the text
    tokens = word_tokenize(msg)
    # Keep only alphanumeric words (remove punctuation)
    word_token = [w for w in tokens if w.isalnum()]
    # Remove stopwords
    clean_token = [w for w in word_token if w not in stop]
    # Lemmatization to get the base form of words
    lemma = WordNetLemmatizer()
    clean_token = [lemma.lemmatize(w) for w in clean_token]
    return ' '.join(clean_token)

# Apply the clean_text function to the 'Text' column
df['Text'] = df['Text'].apply(clean_text)

# Encode the labels (Class)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# Split into input (X) and output (Y)
X = df['Text']  # input
Y = df['Class']  # output

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
print('Training set size:', X_train.shape, Y_train.shape)
print('Testing set size:', X_test.shape, Y_test.shape)

# Check the first sentence in the training set
print(X_train[0])

# Calculate sentence lengths and add as a new column
L = [len(word_tokenize(sent)) for sent in df['Text']]  # List comprehension for sentence length
df['Text'] = L

# Print the maximum sentence length in the dataset
print(max(df['Text']))

# Use the 95th percentile of sentence length to limit sequence length
max_len = int(np.quantile(df['Text'], 0.95))

# Tokenize the text
from tensorflow.keras.preprocessing.text import Tokenizer
token = Tokenizer(oov_token='<nothing>')  # OOV token for out-of-vocabulary words

# Fit the tokenizer on the training data
token.fit_on_texts(X_train)

# Convert text to sequences
sequence_X_train = token.texts_to_sequences(X_train)
print(sequence_X_train[2])

# Create the LSTM model
from tensorflow.keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import Adam

L = len(token.index_word)  # Get the size of the vocabulary

# One-hot encode the labels
from tensorflow.keras.utils import to_categorical
Y_train_one_hot = to_categorical(Y_train, num_classes=class_legnth)

# Create the Sequential model
model = Sequential()
model.add(Embedding(L + 1, 300, input_length=max_len, mask_zero=True, input_shape=(max_len,)))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=64, activation='tanh')))
model.add(Dense(units=32, activation='relu'))  # Hidden layer
model.add(BatchNormalization())
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Dense(units=class_legnth.shape[0], activation='softmax'))  # Output layer with softmax
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Use EarlyStopping to avoid overfitting
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(sequence_X_train, Y_train_one_hot, epochs=15, validation_split=0.2, batch_size=64, callbacks=[early_stopping])

# Convert test data into sequences
sequence_X_test = token.texts_to_sequences(X_test)

# Pad the test sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
sequence_test = pad_sequences(sequence_X_test, padding='post', maxlen=max_len)

# Predict on the test data
Y_pred = model.predict(sequence_test)

# Get the predicted labels
Y_pred_labels = np.argmax(Y_pred, axis=1).reshape(-1, 1)
print(Y_pred_labels)

# Convert the predicted labels back to class names
class_names = label_encoder.inverse_transform(Y_pred_labels)

# Print the predicted class names
predicted_class_names = [class_names[label] for label in Y_pred_labels.flatten()]
print(predicted_class_names)

# Create a mapping of class labels to class names
classes = label_encoder.classes_
class_mapping = {idx: cls for idx, cls in enumerate(classes)}
print(class_mapping)

# Save the class mapping to a file
with open("labels.pkl", 'wb') as file:
    pickle.dump(class_mapping, file)

# Evaluate the model performance
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test, Y_pred_labels))
print(confusion_matrix(Y_test, Y_pred_labels))

# Save the tokenizer and model
file2 = open("token.pkl", 'wb')
pickle.dump(token, file2)
file2.close()

# Save the trained model
model.save('model_300_15.h5')
print("***********************************Model Saved***********************************")
