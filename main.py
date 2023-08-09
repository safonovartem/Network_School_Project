import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import keras
import sklearn
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Bidirectional

df = pd.read_csv('data.csv')

x = df[['information']]
y = df['label']

messages = x.copy()
messages.reset_index(inplace=True)# добавили индекс
target = y.copy()

ps = PorterStemmer()
sw = stopwords.words('english')

corpus= []
fake = []
not_fake = []
for i in tqdm(range(0,len(messages))):
  review = messages['information'][i]
  review = re.sub('[^a-zA-Z]',' ',review)
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if word not in stopwords.words('english')]

  review = ' '.join(review)
  if target[i] == 0:
    not_fake.append(review)
  else:
    fake.append(review)
  corpus.append(review)

voc_size = 20000
onehot_repr = [one_hot(words,voc_size) for words in corpus]
sent_length = len(max(onehot_repr, key = len)) # длина максимального ohehotа
embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length) # уравниваем все onehotы
x_final = np.array(embedded_docs)
y_final = np.array(y) # 0 1

x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, stratify = y_final, test_size = 0.25, random_state = 42)
embedding_vector_features = 30
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.5)) # прореживание
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=50,epochs=10)
results = model.evaluate(x_test,y_test,batch_size=50)
prediction = model.predict(x_test[0:3])