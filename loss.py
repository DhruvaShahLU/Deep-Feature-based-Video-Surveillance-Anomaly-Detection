import torch
import torch.nn.functional as F

# Define a function that takes in the predicted values y_pred, batch size, and an optional is_transformer flag
def MIL(y_pred, batch_size, is_transformer=0):

    # If is_transformer is not set, reshape the tensor to have batch_size rows and an arbitrary number of columns
    # Otherwise, apply the sigmoid function to the tensor
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    # Generate random indices for the anomaly and normal samples
    anomaly_index = torch.randperm(30, device=y_pred.device)
    normal_index = torch.randperm(30, device=y_pred.device)

    # Slice the y_pred tensor into two parts: y_anomaly and y_normal
    # y_anomaly contains the first 32 columns shuffled according to the anomaly_index
    # y_normal contains the rest of the columns shuffled according to the normal_index
    y_anomaly = y_pred[:, :32][:, anomaly_index]
    y_normal = y_pred[:, 32:][:, normal_index]

    # Compute the maximum value along each row of y_anomaly and y_normal
    y_anomaly_max = torch.amax(y_anomaly, dim=1)
    y_normal_max = torch.amax(y_normal, dim=1)

    # Compute the loss function using the maximum values
    # The loss function is defined as the mean of the ReLU function applied to 1 - y_anomaly_max + y_normal_max
    # The second term in the loss function is the sum of two terms: the sum of y_anomaly (scaled by a hyperparameter lambda)
    # and the sum of the squared differences between adjacent columns in the first 31 columns of y_pred (also scaled by lambda too)
    # The entire second term is then divided by the batch size
    leaky_relu = torch.nn.ReLU()
    loss = torch.mean(leaky_relu(1. - y_anomaly_max + y_normal_max)) + \
           torch.mean(torch.sum(y_anomaly, dim=1) * 0.00008 + \
           torch.sum((y_pred[:, :31] - y_pred[:, 1:32]) ** 2, dim=1) * 0.00008) / batch_size

    return loss



