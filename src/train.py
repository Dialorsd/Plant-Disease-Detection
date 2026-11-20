import json
from src import config as C
from src.data import load_datasets
from src.model import build_model


def main():
    train, val, classes = load_datasets()
    model = build_model(len(classes))
    model.fit(train, validation_data=val, epochs=C.EPOCHS)
    C.MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(C.MODEL_PATH)
    C.CLASSES_PATH.write_text(json.dumps(classes))


if __name__ == "__main__":
    main()
