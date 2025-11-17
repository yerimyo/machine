import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
# 합성곱 신경망의 시각화

# 가중치 시각화
import keras
model = keras.models.load_model('best-cnn-model.keras')

model.layers

conv = model.layers[0]