import json
from pathlib import Path

from models import NaiveSimpleNeuralNetwork, MatrixSimpleNeuralNetwork

def main(num_epochs: int = 3, learning_rate: float = 0.1, model_type: str = "matrix"):
    # load data_set
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path, "r") as f:
        data_set = json.load(f)

        samples        = data_set["data"]
        index_to_label = data_set["index_to_label"]
    
    # create model
    if model_type == "naive":
        model = NaiveSimpleNeuralNetwork(num_inputs=len(samples[0]["inputs"]), num_outputs=len(index_to_label))
    elif model_type == "matrix":
        model = MatrixSimpleNeuralNetwork(num_inputs=len(samples[0]["inputs"]), num_outputs=len(index_to_label))

    # train model
    for _ in range(num_epochs):
        for sample in samples:
            loss = model.train(sample["inputs"], sample["label"], learning_rate)

            accuracy = 0
            for sample in samples:
                predicted_label = model.predict(sample["inputs"]).index(max(model.predict(sample["inputs"])))
                accuracy += 1 if predicted_label == sample["label"] else 0

            print(f"Loss: {loss:.4f}, Accuracy: {accuracy / len(samples) * 100:.2f}%")

if __name__ == "__main__":
    main(num_epochs=10, learning_rate=0.1)
