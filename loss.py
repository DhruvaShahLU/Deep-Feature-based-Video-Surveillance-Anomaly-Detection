import torch
import torch.nn.functional as F



def MIL(y_pred, batch_size, is_transformer=0):
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    anomaly_index = torch.randperm(30, device=y_pred.device)
    normal_index = torch.randperm(30, device=y_pred.device)

    y_anomaly = y_pred[:, :32][:, anomaly_index]
    y_normal = y_pred[:, 32:][:, normal_index]

    y_anomaly_max = torch.amax(y_anomaly, dim=1)
    y_normal_max = torch.amax(y_normal, dim=1)
    
    #loss function
  
    relu = torch.nn.ReLU()
    loss = torch.mean(relu(1. - y_anomaly_max + y_normal_max)) + \
           torch.mean(torch.sum(y_anomaly, dim=1) * 0.00008 + \
           torch.sum((y_pred[:, :31] - y_pred[:, 1:32]) ** 2, dim=1) * 0.00008) / batch_size

    return loss
