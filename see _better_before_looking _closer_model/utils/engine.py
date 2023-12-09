
import time
from utils import calculate_pooling_center_loss, mask2bbox
from utils import attention_crop, attention_drop, attention_crop_drop
from utils import getDatasetConfig, getConfig, getLogger
from utils import accuracy, get_lr, save_checkpoint, AverageMeter, set_seed
#from google.colab import files
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
class Engine():
    def __init__(self,):
        pass

    def train(self,state,epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        config = state['config']
        print_freq = config.print_freq
        model = state['model']
        criterion = state['criterion']
        optimizer = state['optimizer']
        train_loader = state['train_loader']
        model.train()
        end = time.time()
        for i, (img, label) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = label.cuda()
            input = img.cuda()
            # compute output
            attention_maps, raw_features, output1 = model(input)
            features = raw_features.reshape(raw_features.shape[0], -1)

            feature_center_loss, center_diff = calculate_pooling_center_loss(
                features, state['center'], target, alfa=config.alpha)

            # update model.centers
            state['center'][target] += center_diff

            # compute refined loss
            # img_drop = attention_drop(attention_maps,input)
            # img_crop = attention_crop(attention_maps, input)
            img_crop, img_drop = attention_crop_drop(attention_maps, input)
            _, _, output2 = model(img_drop)
            _, _, output3 = model(img_crop)

            loss1 = criterion(output1, target)
            loss2 = criterion(output2, target)
            #loss3 = criterion(output3, target)

            loss = (loss1+loss2)/2 + feature_center_loss
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output1, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 4 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    .format(
                        epoch, i, len(train_loader),
                        loss=losses, top1=top1,))
                print("loss1,loss2,feature_center_loss", loss1.item(), loss2.item(),
                    feature_center_loss.item())
        
        return top1.avg, losses.avg
    
    def validate(self,state):
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        config = state['config']
        print_freq = config.print_freq
        model = state['model']
        val_loader = state['val_loader']
        criterion = state['criterion']
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()
                # forward
                attention_maps, raw_features, output1 = model(input)
                features = raw_features.reshape(raw_features.shape[0], -1)
                feature_center_loss, _ = calculate_pooling_center_loss(
                    features, state['center'], target, alfa=config.alpha)

                img_crop, img_drop = attention_crop_drop(attention_maps, input)
                # img_drop = attention_drop(attention_maps,input)
                # img_crop = attention_crop(attention_maps,input)
                _, _, output2 = model(img_drop)
                _, _, output3 = model(img_crop)
                loss1 = criterion(output1, target)
                loss2 = criterion(output2, target)
                loss3 = criterion(output3, target)
                # loss = loss1 + feature_center_loss
                loss = (loss1+loss2)/2+feature_center_loss
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output1, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 1 == 0:
                    print('Test: [{0}/{1}]\t'
                        
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                        .format(
                            i, len(val_loader), loss=losses,
                            top1=top1, ))

            print('  Validation : Accuracy1 {top1.avg:.3f} Loss : {losses.avg:.3f} '
                .format(top1=top1,losses=losses))

        return top1.avg, losses



if __name__ == '__main__':

    engine = Engine()
    engine.train()
