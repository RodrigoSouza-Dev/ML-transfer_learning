pip install tensorflow transformers scikit-learn

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dados de exemplo (sentimentos de análise de filme)
texts = ["Eu adorei esse filme!", "Terrível, não recomendaria a ninguém.", "Um filme mediano, nada de especial." ]
labels = [1, 0, 1]  # 1 para positivo, 0 para negativo

# Divide os dados em conjuntos de treinamento e teste
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Carrega o tokenizer do BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokeniza os textos e converte para tensores
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64, return_tensors='tf')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64, return_tensors='tf')

# Converte as listas de labels para tensores
train_labels = tf.convert_to_tensor(train_labels)
test_labels = tf.convert_to_tensor(test_labels)

# Carrega o modelo pré-treinado BERT para classificação de sequência
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compila o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Treina o modelo com os dados
model.fit(train_encodings, train_labels, epochs=3, batch_size=8)

# Avalia o modelo com os dados de teste
loss, accuracy = model.evaluate(test_encodings, test_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Faz predições com os dados de teste
predictions = model.predict(test_encodings)
predicted_labels = np.argmax(predictions.logits, axis=1)

# Imprime o relatório de classificação
print(classification_report(test_labels, predicted_labels))

