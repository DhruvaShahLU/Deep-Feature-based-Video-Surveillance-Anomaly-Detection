from torch.utils.data import DataLoader
# from learner2 import *
from learner import *
# from learner2 import HOE_model
from loss import MIL
# from loss_2 import MIL2
from dataset import *
import os
from sklearn import metrics
import argparse
# from FFC import *
import matplotlib.pyplot as plt
import torch.nn as nn
normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Learner(input_dim=2048, drop_p=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay=0.0010000000474974513)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL
criterion2 = nn.BCELoss()
criterion3 = nn.BCELoss()



def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        #  print('normal_inputs shape:', normal_inputs.shape) # torch.Size([30, 32, 2048])
        # print(normal_inputs[0])
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        #  print('inputs shape:', inputs.shape) # torch.Size([30, 64, 2048])
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs) # torch.Size([1920, 1]) 
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # train_losses.append(train_loss)
    print('loss = ', train_loss/len(normal_train_loader))
    training_loss = train_loss/len(normal_train_loader)
    length = str(train_loss/len(normal_train_loader))
    # print(len(length))
    scheduler.step()
    return training_loss

def test_abnormal(epoch):
    model.eval()
    auc = 0
    total_loss = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data 
            # [1,312,2048]
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)

            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, torch.div(frames, 16), 33))
            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, torch.div(frames, 16), 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            # print(gt_list3)

            # Convert arrays to PyTorch tensors
            gt_tensor = torch.from_numpy(gt_list3).float().to(device)
            score_tensor = torch.from_numpy(score_list3).float().to(device)
            batch_size = 1
            # print(gt_tensor)
            

            # Calculate loss
            test_loss = criterion2(score_tensor, gt_tensor)
            # t_loss = criterion(gt_tensor,batch_size)
            # test_loss = t_loss/len(anomaly_test_loader)
            

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)

        # test_losss = test_loss / len(anomaly_test_loader)
        test_losss = test_loss.item()
        print('auc = ', auc/140)
        print('test loss =', test_losss)
        return test_losss/len(normal_train_loader), auc/140




if __name__ == '__main__':
    train_lossess = []
    test_lossess = []
    best_auc = 0
    for epoch in range(1, 101):
        train_l= train(epoch)
        test_l,auc = test_abnormal(epoch)
        # test_l = train_l * 0.00003
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join('best_model.pth'))
        train_lossess.append(train_l)
        test_lossess.append(test_l)

    # model_dir = "/content/drive/MyDrive/Project/"   
    # # torch.save(model.state_dict(), os.path.join('final_model.pth'))
    # torch.save(model.state_dict(), os.path.join(model_dir, "final_model3.pth"))
    print("best AUC = ", best_auc)      

# for epoch in range(0, 75):
#     train(epoch)
#     test_abnormal(epoch)
