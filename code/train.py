from model import build_cnn

model = build_cnn(input_shape=(28, 28, 1), num_classes=10)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
