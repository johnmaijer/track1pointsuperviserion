import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import  Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

parser = argparse.ArgumentParser(description="PyTorch LESPS train")
parser.add_argument("--model_names", default=['SCTransNet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'SCTransNet'")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'NUDT-SIRST-Sea', 'SIRST3'")
parser.add_argument("--label_type", default='full', type=str, help="label type: centroid, coarse")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets/', type=str, help="train_dataset_dir, default: './datasets/")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch sizse, default: 16")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size, default: 256")
parser.add_argument("--save", default='./log', type=str, help="Save path, default: './log")
parser.add_argument("--resume", default=None, type=str, help="Resume path, default: None")

parser.add_argument("--nEpochs", type=int, default=600, help="Number of epochs, default: 1000 for SCTransNet")
parser.add_argument("--logEnpoch", type=int, default=10, help="print the train loss")
parser.add_argument("--test_iter", type=int, default=10, help="训练多少轮在测试集上测试一次，并保存最佳权重")

parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate, default: 5e-4")
parser.add_argument("--optimizer_name", type=str, default='Adam', help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma, default: 0.1')
parser.add_argument("--step", type=int, default=[200, 300], help="Sets the learning rate decayed by step, default: [200, 300]")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, default: 1")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test, default: 0.5")
parser.add_argument("--cache", default=False, type=str, help="True: cache intermediate mask results, False: save intermediate mask results")


global opt
opt = parser.parse_args()
def train():
    train_set = TrainSetLoader_full(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batchSize, shuffle=True)
    
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    best_source = 0
    best_mIOU = 0
    best_idx_epoch = 0
    best_Pd = 0
    net = Net(model_name=opt.model_name, mode='train').cuda()

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
        for i in range(len(opt.step)):
            opt.step[i] = opt.step[i] - epoch_state

    if opt.model_names == 'SCTransNet':
        net.apply(weights_init_kaiming)
        ### Default settings of SCTransNet
        if opt.optimizer_name == 'Adam':
            opt.optimizer_settings = {'lr': 1e-3}  # 可以尝试调整学习率看看你 1e-4~1e-6
            opt.scheduler_name = 'CosineAnnealingLR'
            opt.scheduler_settings = {'epochs': opt.nEpochs, 'eta_min': 1e-5, 'last_epoch': -1}

        ### Default settings of DNANet
        if opt.optimizer_name == 'Adagrad':
            opt.optimizer_settings = {'lr': 0.05}
            opt.scheduler_name = 'CosineAnnealingLR'
            opt.scheduler_settings = {'epochs': opt.nEpochs, 'min_lr': 1e-5}

        ### Default settings of EGEUNet
        if opt.optimizer_name == 'AdamW':
            opt.optimizer_settings = {'lr': 0.001, 'betas': (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2,
                                      "amsgrad": False}
            opt.scheduler_name = 'CosineAnnealingLR'
            opt.scheduler_settings = {'epochs': opt.nEpochs, 'T_max': 50, 'eta_min': 1e-5, 'last_epoch': -1}
        opt.nEpochs = opt.scheduler_settings['epochs']

        optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                             opt.scheduler_settings)

    else:
       optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
       scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.step, gamma=opt.gamma)
    
    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            net.train()
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask, opt.model_name)
            total_loss_epoch.append(loss.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        scheduler.step()
        '''
        print each epoch loss and total loss 
        '''
        if (idx_epoch + 1) % (opt.logEnpoch) == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
        '''
        every test_iter epoch train, test on the test_dataset  get the best result 
        '''
        if (idx_epoch + 1) % opt.test_iter == 0 or  (idx_epoch + 1) == opt.nEpochs:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.save_perdix + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                'train_iou_list': opt.train_iou_list,
                'test_iou_list': opt.test_iou_list,
            }, save_pth)
            result1, result2, source = test(save_pth)
            if source > best_source:
                best_source = source
                best_mIOU = result1[1] * 100
                best_Pd = result2[0] * 100
                best_idx_epoch = idx_epoch + 1
            print("when got the best source, the best mIOU:", best_mIOU, 'best_pd:', best_Pd, "at epoch ", best_idx_epoch, "with source ", best_source)

            
def test(save_pth):
    '''

    Args:
        save_pth:
    Returns:
        result1(tuple): (pixAcc, mIoU)
        result2(tuple): (PD, FA)
    '''
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)

    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()

    #########################
    # 按照比赛要求设计模型得分
    source  = get_model_sources(results1, results2)
    if source == 0:
        print("当前epoch的FA过高,大于1e-4")
    else:
        print(f"模型的得分为:{source}")
    ############################
    print("Inference mask pixAcc, mIoU:\t" + str(results1))
    print("Inference mask PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    return results1, results2, source

def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            opt.save_perdix = opt.model_name + opt.label_type

            ### save intermediate loss vaules
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + opt.label_type + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            opt.train_iou_list = []
            opt.test_iou_list = []
            
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
