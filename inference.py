import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
from tqdm import tqdm
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch LESPS test")
parser.add_argument("--model_names", default=['SCTransNet'], nargs='+', help="model_name: 'ACM', 'ALCNet', 'DNANet'")
parser.add_argument("--pth_dirs", default=['NUAA-SIRST/best.pth.tar'],
                                            nargs='+', help="checkpoint dir, default=None")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets/', type=str, help="train_dataset_dir")
parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size, default: 512")
parser.add_argument("--post_process", type=str, default='MultiThresholds', help="the post process for the output of the net. Default: MultiThresholds, corrode")

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

def test(): 
    test_set = InferenceSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, size, img_dir) in tqdm(enumerate(test_loader)):
            # img = Variable(img).cuda()
            img = img.cuda()
            b, c, h, w = img.shape
            # 当输入图像大于512的时候
            if h > opt.patchSize and w > opt.patchSize:
                img_unfold = F.unfold(img[:,:,:,:], opt.patchSize, stride=opt.patchSize)
                img_unfold = img_unfold.reshape(c, opt.patchSize, opt.patchSize, -1).permute(3, 0, 1, 2)
                patch_num = img_unfold.size(0)
                for pi in range(patch_num):
                    img_pi = img_unfold[pi, :,:,:].unsqueeze(0).float()
                    # img_pi = Variable(img_pi)
                    preds_pi = net.forward(img_pi)
                    if pi == 0:
                        preds = preds_pi
                    else:
                        preds = torch.cat([preds, preds_pi], dim=0)
                preds = preds.permute(1, 2, 3, 0).unsqueeze(0)
                pred = F.fold(preds.reshape(1,-1,patch_num), kernel_size=opt.patchSize, stride=opt.patchSize, output_size=(h,w))
            else: 
                pred = net.forward(img)  
            pred = pred[:,:,:size[0],:size[1]]
            '''
            计算模型输出的整体情况
            '''
            # topk_pred_values, _ = pred.squeeze().contiguous().view(-1).topk(20)
            # topk_avg_list.append(topk_pred_values.cpu().detach().numpy().sum() / 20)
            # topk_values_list.append(topk_pred_values)
            # 将预测结果转换为多个二值化图像
            '''
            多重阈值融合分割优化
            '''
            if opt.post_process == 'MultiThresholds':
                   thresholds = [0.2, 0.4, 0.6, 0.8]  # 多级阈值
                   binary_masks = []
                   for threshold in thresholds:
                       pred_binary = (pred >= threshold).int()
                       binary_masks.append(pred_binary.squeeze().cpu().detach().numpy().astype(np.uint8) * 255)
                   # 组合多个二值化图像进行逐步细化
                   refined_mask = binary_masks[0]
                   for binary_mask in binary_masks[1:]:
                       refined_mask = cv2.bitwise_or(refined_mask, binary_mask)
                   pred_binary = refined_mask
                   # 将细化后的结果转换为张量
                   # refined_mask_tensor = torch.from_numpy(refined_mask).unsqueeze(0).unsqueeze(0).to(pred.device)
            elif opt.post_process == 'corrode':
                   pred_binary = (pred >= opt.threshold).int() #  0.3393620839349927根据模型输出信息，做的一个topk均值处理
                   non_zero_indices = pred_binary.nonzero()
                   num_non_zero = non_zero_indices.size(0)  # 获取非零元素的数量
                   # 腐蚀操作
                   if num_non_zero >=  49.89 * 1.2: # 49.89 数据集中综合所有样本中小样本像素的均值, 1.2超参
                        eroded_operate_kernel = np.ones((1, 2), dtype=np.uint8)
                        pred_binary = cv2.erode(pred_binary.squeeze().cpu().detach().numpy().astype(np.uint8)* 255, eroded_operate_kernel, iterations=1)
                   else:
                     pred_binary = pred_binary.squeeze().cpu().detach().numpy().astype(np.uint8)* 255
            else:
                pred_binary = pred
            # pred_binary = torch.from_numpy(np.ascontiguousarray(pred_binary)).unsqueeze(0).unsqueeze(0).to(pred.device)
            ### save img
            model_name = opt.pth_dir.split('/')[-1] .split('.')[0]
            if opt.save_img == True:
                if isinstance(pred_binary, np.ndarray):
                   img_save = Image.fromarray(pred_binary)
                elif isinstance(pred_binary, torch.Tensor):
                   img_save = transforms.ToPILImage()((pred[0, :, :size[0], :size[1]]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + model_name ):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + model_name )
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + model_name + '/' + img_dir[0] + '.png')
                              
if __name__ == '__main__':
    for pth_dir in opt.pth_dirs:
        opt.train_dataset_name = pth_dir.split('/')[0]
        print(pth_dir)
        for dataset_name in opt.dataset_names:
            opt.test_dataset_name = dataset_name
            opt.pth_dir = opt.save_log + pth_dir
            print(opt.test_dataset_name)
            for model_name in opt.model_names:
                    opt.model_name = model_name
            test()
        print('\n')
