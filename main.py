from torch.utils.data import DataLoader
from FCNN import *
from loss import MIL
from dataset import *
import os
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Deep Feature-based Anomaly Detection for Video Surveillance')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
parser.add_argument('--modality', default='TWO', type=str, help='modality')
parser.add_argument('--input_dim', default=2048, type=int, help='input_dim')
parser.add_argument('--drop', default=0.1, type=float, help='dropout_rate')
args = parser.parse_args()

best_auc = 0

normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality)
normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality)

anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality)
anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
len_normal_loader = 137
anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
MIL2 = nn.BCELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FCNN(input_dim=args.input_dim).to(device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.w)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL
criterion2 = MIL2 

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
#         inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        
#         batch_size = inputs.shape[0]
#         inputs = inputs.view(-1, inputs.size(-1)).to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, batch_size)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     print('loss = ', train_loss/len(anomaly_train_loader)) # use length of anomaly_train_loader
#     training_loss = train_loss/len(anomaly_train_loader) # use length of anomaly_train_loader
#     scheduler.step()
#     return training_loss
# Define function named "train" which takes an argument called "epoch"
def train(epoch):
    # Print a string with the "epoch" variable included
    print('\nEpoch: %d' % epoch)
    
    # Put the model in training mode
    model.train()
    
    # Initialize variables to track the training loss, correct predictions, and total predictions
    train_loss = 0
    correct = 0
    total = 0
    
    # Loop through the batches in both the normal and anomaly train loaders
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        # Concatenate the normal and anomaly inputs along the batch dimension
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        
        # Get the batch size and flatten the inputs
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        
        # Pass the inputs through the model and calculate the loss
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        
        # Zero out the gradients, backpropagate the loss, and take a step using the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Add the loss to the running total
        train_loss += loss.item()
        
    # Print the average loss over the entire dataset
    print('loss = ', train_loss/len(anomaly_train_loader)) # use length of anomaly_train_loader
    
    # Calculate and store the training loss over the entire dataset
    training_loss = train_loss/len(anomaly_train_loader) # use length of anomaly_train_loader
    
    # Adjust the learning rate using the scheduler
    scheduler.step()
    
    # Return the training loss
    return training_loss

def test_abnormal(epoch):
    model.eval()  # set model to evaluation mode
    auc = 0  # initialize auc to 0
    total_loss = 0  # initialize total loss to 0
    with torch.no_grad():  # disable gradient calculation for faster inference
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)): # loop over anomaly and normal test data loaders
            inputs, gt, frames = data  # get input data, ground truth labels and number of frames from the anomaly test data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))  # reshape and move inputs to GPU
            
            sc = model(inputs)  # get anomaly score for the input
            
            sc = sc.cpu().detach().numpy()  # move score to CPU and convert it to a numpy array
            sc_list = np.zeros(frames[0])  # initialize score list to an array of zeros with size equal to the number of frames in the input
            
            level = np.round(np.linspace(0, torch.div(frames, 16), 33))  # divide frames into 33 equal parts and calculate the indices for each part
            for j in range(32):  # loop over each part
                sc_list[int(level[j])*16:(int(level[j+1]))*16] = sc[j]  # assign the anomaly score to the corresponding indices in the score list

            gt_list = np.zeros(frames[0])  # initialize the ground truth label list to an array of zeros with size equal to the number of frames in the input
            for j in range(len(gt)//2):  # loop over the ground truth labels
                s = gt[j*2]  # get the start index for the ground truth label
                e = min(gt[j*2+1], frames)  # get the end index for the ground truth label, taking into account the maximum number of frames
                gt_list[s-1:e] = 1  # set the corresponding indices in the ground truth label list to 1

            inputs2, gt2, frames2 = data2  # get input data, ground truth labels and number of frames from the normal test data
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))  # reshape and move inputs to GPU
            sc2 = model(inputs2)  # get anomaly score for the input
            sc2 = sc2.cpu().detach().numpy()  # move score to CPU and convert it to a numpy array
            sc_list2 = np.zeros(frames2[0])  # initialize score list to an array of zeros with size equal to the number of frames in the input
            level2 = np.round(np.linspace(0, torch.div(frames, 16), 33))  # divide frames into 33 equal parts and calculate the indices for each part
            for j2 in range(32):  # loop over each part
                sc_list2[int(level2[j2])*16:(int(level2[j2+1]))*16] = sc2[j2]  # assign the anomaly score to the corresponding indices in the score list
            gt_list2 = np.zeros(frames2[0])  # initialize the ground truth label list to an array of zeros with size equal to the number of frames in the input
            sc3 = np.concatenate((sc_list, sc_list2), axis=0)  # concatenate the anomaly score lists from both anomaly and normal test data
           

            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0) # concatenating two arrays along axis 0

            # Convert arrays to PyTorch tensors
            gt_tensor = torch.from_numpy(gt_list3).float() # converting numpy array to PyTorch tensor
            sc_tensor = torch.from_numpy(sc3).float() # converting numpy array to PyTorch tensor
            batch_size = 1

            # Calculate loss
            test_loss = criterion2(sc_tensor, gt_tensor) # calculating loss using the criterion2

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, sc3, pos_label=1) # calculating the false positive rate, true positive rate, and threshold using the roc_curve function from scikit-learn's metrics module
            auc += metrics.auc(fpr, tpr) # calculating AUC using the auc function from scikit-learn's metrics module

    test_losss = test_loss.item() # getting the scalar value of the loss using the item function
    print('auc = ', auc/len_normal_loader) # printing the average AUC over the entire dataset
    print('test loss =', test_losss) # printing the loss
    return test_losss/len(anomaly_test_loader), auc/len_normal_loader # returning the normalized loss and AUC


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Initializes the EarlyStopper object with the given values of patience and min_delta.

        Args:
            patience (int): The number of epochs to wait before stopping if the validation loss does not improve.
            min_delta (float): The minimum change in the validation loss to be considered as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """
        Implements the early stopping functionality by checking if the validation loss has improved.

        Args:
            validation_loss (float): The validation loss of the current epoch.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if __name__ == '__main__':
    train_lossess = []
    test_lossess = []
    best_auc = 0
    All_AUC = []
    # Create an instance of the EarlyStopper class
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    # Train the model for 100 epochs
    for epoch in range(1, 101):
        # Compute the training loss for the current epoch
        train_l = train(epoch)
        # Compute the test loss and AUC (Area Under the Curve) for the current epoch
        test_l, auc = test_abnormal(epoch)
        # Check if early stopping condition is met based on the training loss
        if early_stopper.early_stop(train_l):
            break
        # If the current AUC is higher than the previous best, update the best AUC and save the model parameters
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join('best_model.pth'))
        # Append the training and test losses to their corresponding lists
        train_lossess.append(train_l)
        test_lossess.append(test_l)
        All_AUC.append(auc)

    # Print the best AUC seen during training
    print("best AUC = ", best_auc)






# Define the range of epochs from 1 to 100
epochs = range(1, 101)

# Plot the training loss as a function of epoch
plt.plot(epochs, train_lossess, 'b', label='Training loss')
plt.title('Loss vs Epochs')
plt.legend(['Training Loss'])
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

# Create a figure with two subplots, sharing the x-axis
fig, ax1 = plt.subplots()

# Plot testing AUC on the left axis
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('AUC', color=color)
ax1.plot(epochs, All_AUC, 'b', label='Testing AUC')
ax1.tick_params(axis='y', labelcolor=color)

# Create a twin axis for testing loss on the right
ax2 = ax1.twinx()

# Plot testing loss on the right axis
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(epochs, test_lossess, 'r', label='Testing loss')
ax2.tick_params(axis='y', labelcolor=color)

# Add a legend for testing AUC
ax1.legend(loc='upper left')

# Add a legend for testing loss
ax2.legend(loc='upper right')

# Set the title of the figure
plt.title('AUC and Loss vs Epochs')
plt.show()

