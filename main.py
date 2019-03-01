from carla_loader import carla
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.backends import cudnn
from cnn_model import Model
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from random import randint
cudnn.benchmark=False
writer = SummaryWriter(comment='code')

args = {
'train_batch_size':4,
'val_batch_size': 4,
'lr':1e-4,
'num_worker': 24,
'max_epoch':100,
'device1':'cuda:1',
'device2':'cuda:0',
'exp':1,
'regression': False,
}

def main(args):
    model = Model(args['train_batch_size'],256,512,4,num_class = 13).to(args['device1']) 

    print('training with 4 images as input and predicting the fifth one in carla with regression')
    train_set = carla(mode = 'train')
    val_set = carla(mode = 'val')

    train_loader = DataLoader(train_set,batch_size = args['train_batch_size'],num_workers = args['num_worker'],shuffle = False, pin_memory=True) 
    val_loader = DataLoader(val_set,batch_size = args['val_batch_size'],num_workers = args['num_worker'],shuffle = False, pin_memory=True)
#end of loading data     
    print(train_loader.__len__())
    if args['regression']:
       criterion = gdl_loss_L1_Loss().to(args['device1'])
       
    else:
        criterion = nn.CrossEntropyLoss(ignore_index = 0).to(args['device1'])
    optimizer = optim.Adam(model.parameters(),lr =args['lr'],betas = (0.9,0.999))
    train(model,criterion,optimizer,train_loader,val_loader)

def train(model,criterion,optimizer,train_loader,val_loader):
    model.train()
    vi = 0
    total_loss = 0.0
#    epoch = -1
#    validate(model,epoch,criterion,val_loader,optimizer)
    for epoch in tqdm(range(args['max_epoch']),'epoch: '):
        for data_future,speed_future,frames,gt in tqdm(train_loader,'iter: '):
            frames = Variable(frames).to(args['device1']).float()
            gt = Variable(gt).to(args['device1']).long()
            data_future = Variable(one_hot(data_future)).to(args['device1'])
            #frame_vis = vutils.make_grid(carla.colorize_mask(frames[0][0].data.cpu().numpy()))
            output = model(frames,data_future,speed_future)
#            output_vis = vutils.make_grid(carla.colorize_mask(output.max(1)[1][0].data.cpu().numpy()))
#            gt_vis = vutils.make_grid(carla.colorize_mask(gt[0].data.cpu().numpy()))
#            final_vis = torch.cat((gt_vis,output_vis),2)
#            writer.add_image('final',final_vis,vi)
            if args['regression']:
                #output = output.max(1)[1].unsqueeze(1).float()
                gt = gt.unsqueeze(1).float()
                loss = criterion(output,gt) 
            else:
                loss = criterion(output,gt)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            vi+=1
            writer.add_scalar('train_loss',loss.item(),vi)
        print('-------------------------------------')
        print('train loss: %f'%(total_loss/vi))
        print('-------------------------------------')
        validate(model,epoch,criterion,val_loader,optimizer)

def validate(model,epoch,criterion,val_loader,optimizer):
    model.eval()
    vi = 0
    total_loss = 0.0
    predictions_all = []
    gts_all = []
    #data_all = torch.zeros([8,1,180],dtype = torch.int32).to(args['device1'])

    for data_future,speed_future,frames,gt in tqdm(val_loader,'iter: '):
        with torch.no_grad():
            frames = Variable(frames).to(args['device1']).float()
            gt = Variable(gt).to(args['device1']).long()
            data_future = Variable(one_hot(data_future)).to(args['device1'])
            output = model(frames,data_future,speed_future)

            output_vis = vutils.make_grid(carla.colorize_mask(output.max(1)[1][0].data.cpu().numpy()))
            gt_vis = vutils.make_grid(carla.colorize_mask(gt[0].data.cpu().numpy()))
            final_vis = torch.cat((gt_vis,output_vis),2)
            writer.add_image('gt/prediction',final_vis,vi)
            if args['regression']:
                #output = output.max(1)[1].unsqueeze(1).float()
                gt = gt.unsqueeze(1).float()
                loss = criterion(output,gt)
            else:
                loss = criterion(output,gt)

        predictions_all.append(output.max(1)[1].cpu().numpy())
        gts_all.append(gt.cpu().numpy())
        total_loss += loss.item()
        vi+=1
    predictions_all = np.concatenate(predictions_all)
    gts_all = np.concatenate(gts_all)
    acc,acc_cls,mean_iu,fwavacc = evaluate(predictions_all,gts_all,19)
    print('mean_iu: %f | val loss: %f'%(mean_iu,total_loss/vi))
    print('-------------------------------------')
    #print(data_all.cpu().numpy())
    #print('-------------------------------------')

    model.save_model(args['exp'],epoch,mean_iu,optimizer)

    model.train()

def one_hot(inp,category = 'degree',num_classes = 180):
    y_onehot = torch.zeros([inp.shape[0],1,num_classes],dtype = torch.float32)
    #if category == 'degree':
    #    for i in range(inp.shape[0]):
    #        inp[i] = inp[i] + 90
    for i in range(inp.shape[0]):
        y_onehot[i,0,:] = torch.eye(num_classes)[int(inp[i])]
    return y_onehot

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate(predictions, gts, num_classes = 19):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


'''
GDL loss for the image reconstruction
Note, this copy of gdl loss is successful. It may not be able to pass the gradcheck, that's becuase of the float/double issue, but it is fine.
If we change the filter to torch.DoubleTensor, and also cast the pred and gt as double tensor, it will pass the grad check. However, float is fine 
with the real application.
By Dong Nie
'''    
class gdl_loss_L1_Loss(nn.Module):   
    def __init__(self, pNorm=2):
        super(gdl_loss_L1_Loss, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)
        
        filterX = Variable(torch.FloatTensor([[[[-1, 1]]]]),requires_grad = False)
        filterY = Variable(torch.FloatTensor([[[[1], [-1]]]]),requires_grad = False)
        #filterX = torch.FloatTensor([[[[-1, 1]]]])  # 1x2
        #filterY = torch.FloatTensor([[[[1], [-1]]]])  # 2x1 
        
        #self.convX.weight = torch.nn.Parameter(filterX,requires_grad=True)
        #self.convY.weight = torch.nn.Parameter(filterY,requires_grad=True)
        self.pNorm = pNorm
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, gt):
        l1 = self.l1_loss(pred,gt)
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())
        
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))
        
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        
        
        mat_loss_x = grad_diff_x ** self.pNorm
        
        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height
        
        shape = gt.shape

        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (shape[0] * shape[1] * shape[2] * shape[3]) 
               
        return mean_loss + l1 
    
main(args)
