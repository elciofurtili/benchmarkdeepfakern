import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Função para gerar y_pred e y_true para um modelo
def get_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Original", "Falsificado"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {
    "MLP": model_mlp.to(device),
    "CNN": model_cnn.to(device),
    "RNN": model_rnn.to(device),
    "LSTM": model_lstm.to(device),
    "GRU": model_gru.to(device)
}

for name, model in models.items():
    y_true, y_pred = get_predictions(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, name)
