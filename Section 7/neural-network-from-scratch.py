from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=4, activation='sigmoid', input_shape=(3,)))

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(units=2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd')
