from keras.layers import Activation, Dense, Dropout  # type: ignore
from keras.models import Model, Sequential  # type: ignore


def create_model(input_size: int) -> Model:
    """Create sequential keras model with `input_size` input size."""
    num_labels = 2

    # Construct model
    model = Sequential()

    model.add(Dense(256, input_shape=(input_size,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )
    return model


def load_trained_model(weights_path: str, input_size: int) -> Model:
    """Create model an load weights from path."""
    model = create_model(input_size)
    model.load_weights(weights_path)
    return model
