from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import sklearn.metrics as metrics
from pycm import *

from keras.layers import Dropout



## setosa, versicolor e virginica

iris = load_iris()
#X = preprocessing.scale(iris['data']) 
X = iris['data']

##função to_categorical transforma os resultados em binários (mais fácil para treinar) | (prática comum)
Y = to_categorical(iris['target'])

## 30 % esta no testes e os outros 70% é o treinamento
#training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

#model
model = Sequential()

#6 neuronios na primeira camada.
#input_dim é 4 pq tem 4 diferentes atributo
model.add(Dense(80, input_dim=4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, input_dim=4, activation='relu'))

#temos valores binarios, neste caso temos 3 saidas.
#usamos o softmax pois é sempre usado para estas classificações de multis-classes  
model.add(Dense(3, activation='softmax'))

# este é o loss que também é usado para classificações de  multi-classes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, batch_size=100)
print(history.history.keys())

#Gráfico Desempenho da accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='uper left')
plt.show()

#Gráfico Desempenho do loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='uper left')
plt.show()

#evaluate model
# _, accuracy = model.evaluate(X_test, Y_test)
# print("Accuracy: %.2f" % (accuracy * 100))


pred = model.predict_classes(X_test)

rounded_labels=np.argmax(Y_test, axis=1)
rounded_labels[1]

#Confusion matrix pycm biblioteca para multi-class
cm = ConfusionMatrix(rounded_labels,pred)

#print(cm)

print("SETOSA, VERSICOLOR e VIRGINICA")
print("---ACCURACY----")
print(cm.ACC)
print("---TPR----")
print(cm.TPR)
print("---TNR----")
print(cm.TNR)


# tn = multilabel_confusion_matrix(rounded_labels, pred).ravel()
# fp = multilabel_confusion_matrix(rounded_labels, pred).ravel()
# fn = multilabel_confusion_matrix(rounded_labels, pred).ravel()
# tp = multilabel_confusion_matrix(rounded_labels, pred).ravel()
# tpr = tp / (tp+fn)
# tnr =  tn / (tn+fp)
# acc = (tp+tn) / (tp+tn+fn+fp)

# print("TPR", tpr)
# print("TNR", tnr)
# print("ACC", acc)

# #AUC

# fpr, tpr, threshold = metrics.roc_curve(Y_test, pred, pos_label=1)
# auc = metrics.auc(fpr, tpr)

# #CODIGO PARA PLOTAR O GRAFICO DA CURVA ROC

# plt.plot([0,1], [0,1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
# plt.plot(fpr,tpr, color='b', label=r'ROC (AUC = %0.2f)' % (auc), lw=2, alpha=.8)
# plt.suptitle("ROC Curve")
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend(loc="lower right")
# plt.show()
