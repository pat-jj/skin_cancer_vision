import torch
from torch.autograd import Variable

from utils import AverageMeter

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    loss_train, acc_train = [],[]

    curr_iter = (epoch - 1) * len(train_loader)

    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        tmp = prediction.eq(labels.view_as(prediction)).sum().item()
        train_acc.update(tmp/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 10 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
            loss_train.append(train_loss.avg)
            acc_train.append(train_acc.avg)
            
    return train_loss.avg, train_acc.avg, loss_train, acc_train

def validate(val_loader, model, criterion, optimizer, epoch, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            tmp = prediction.eq(labels.view_as(prediction)).sum().item()
            val_acc.update(tmp/N)
            val_loss.update(criterion(outputs, labels).item())

    print('-'*50)
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('-'*50)
    return val_loss.avg, val_acc.avg