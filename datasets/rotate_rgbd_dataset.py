import cv2
import os
from torch.utils.data import Dataset,DataLoader
import random
import torch
import numpy as np
from .utils import *
from skimage import io


class RotateRGBDDataset(Dataset):
    def __init__(self,data_file, size=(448, 448),stride=8, aug=True):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.train_data = f.readlines()

        self.size = (320, 320)
        self.aug=aug
        self.stride = stride # for generating gt-mask needed to compute local-feature loss
        self.query_pts = self._make_query_pts()
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)
        self.mean_dep = np.array([0.47880623, 0.47880623, 0.47880623],dtype=np.float32).reshape(3,1,1)
        self.std_dep = np.array([0.20955773, 0.20955773, 0.20955773],dtype=np.float32).reshape(3,1,1)


    def _read_file_paths(self,data_dir):
        assert os.path.isdir(data_dir), "%s should be a dir which contains images only"%data_dir
        file_paths = os.listdir(data_dir)
        return file_paths

    def __getitem__(self, index: int):
        opt, depth  = self.train_data[index].strip('\n').split(' ')
        opt_img_path = os.path.join(os.path.dirname(self.data_file), '', opt)
        opt_img = cv2.imread(opt_img_path.replace('stage1_', ''))
        #pos = np.where(np.max(opt_img[:, :, 0], axis=1) != 0)[0]
        #opt_img = opt_img[pos, ...]
        #heatmapshow = None
        #opt_img = cv2.normalize(opt_img[:,:, 0], heatmapshow, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        opt_img = cv2.cvtColor(opt_img,cv2.COLOR_BGR2RGB)
        #opt_img = cv2.cvtColor(opt_img,cv2.COLOR_GRAY2RGB)
        #opt_img = cv2.copyMakeBorder(opt_img,150,150,150,150,cv2.BORDER_CONSTANT,value=(0,0,0))
        h, w ,c  = opt_img.shape
        #h_ratio = h / 320
        #w_ratio = w / 320
        #opt_img = cv2.resize(opt_img,self.size)
        # opt_img = opt_img[:480, :480]  # 直接裁剪左上角 480x480

        dep_img_path = os.path.join(os.path.dirname(self.data_file), '', depth)
        dep_img = io.imread(dep_img_path.replace('stage1_', ''))
        #dep_img = dep_img.astype(np.float16)
        #dep_img = cv2.cvtColor(dep_img,cv2.COLOR_GRAY2RGB)
        dep_img = np.repeat(dep_img[..., np.newaxis], 3, 2)
        # dep_img = dep_img[:480, :480]  # 直接裁剪左上角 480x480
        #sar_img = sar_img[pos, ...]
        #print(opt_img.shape, dep_img.shape)
        #opt_img = sar_img

        query, refer, Mr, Mq, qc, rc, H_gt = self._generate_ref(opt_img, dep_img)
        # print(query.shape, refer.shape, Mr.shape, Mq.shape)
        # dropout query
        label_matrix = self._generate_label(Mr,Mq, qc, rc, (0, 0)) #400x400
        #print(label_matrix.shape)
        #cv2.imshow("query:", query)
        #cv2.imshow("refer:", refer)
        #cv2.waitKey()
        query = query.transpose(2,0,1)
        refer = refer.transpose(2,0,1)

        query = ((query / 255.0) - self.mean_dep) / self.std_dep
        refer = ((refer / 255.0) - self.mean) / self.std

        sample = {
            "refer":refer,
            "query":query,
            "gt_matrix":label_matrix,
            "h_gt": H_gt,
            # "M": M,
            # "Mr": Mr,
            # "Mq": Mq
        }
        return sample

    def _generate_ref(self, refer, query):
        """
        通过裁剪生成查询图像和参考图像，不进行任何数据增强。
        """
        # 初始化 H_gt
        H_gt = np.eye(3)  # 初始化为单位矩阵

        # 裁剪查询图像
        crop_query, crop_M_query, qc = self._random_crop2(query)
        query = crop_query  # 直接使用裁剪后的图像
        Mq = crop_M_query  # 直接使用裁剪变换矩阵

        # 裁剪参考图像
        crop_refer, crop_M_refer, rc = self._random_crop3(refer)
        refer = crop_refer  # 直接使用裁剪后的图像
        Mr = crop_M_refer  # 直接使用裁剪变换矩阵

        # 计算单应矩阵 H_gt
        H_gt = np.eye(3) + (Mq - Mr)

        return query, refer, Mr, Mq, qc, rc, H_gt

    def _generate_label(self, Mr, Mq, qc, rc, coor, drop_mask=True):
        # print(self.size[0], self.stride)
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        label = np.zeros((ncols * nrows, ncols * nrows))  # (1600, 1600)

        Mq_inv = np.linalg.inv(Mq)
        src_pts = np.matmul(Mq_inv, self.query_pts.T)
        mask0 = (0 <= src_pts[0, :]) & (src_pts[0, :] < self.size[0]) & (0 <= src_pts[1, :]) & (
                    src_pts[1, :] < self.size[1])

        trans_M = np.array([
            [1, 0, coor[0]],
            [0, 1, coor[1]],
            [0, 0, 1]
        ])
        refer_pts = np.matmul(trans_M, src_pts)

        trans_M1 = np.array([
            [1, 0, qc[0]],
            [0, 1, qc[1]],
            [0, 0, 1]
        ])
        trans_M2 = np.array([
            [1, 0, qc[0] - rc[0]],
            [0, 1, qc[1] - rc[1]],
            [0, 0, 1]
        ])
        trans_M = np.matmul(trans_M2, trans_M1)
        trans_M3 = np.array([
            [1, 0, -rc[0]],
            [0, 1, -rc[1]],
            [0, 0, 1]
        ])
        trans_M = np.matmul(trans_M3, trans_M)
        refer_pts = np.matmul(trans_M, refer_pts)
        refer_pts = np.matmul(Mr, refer_pts)

        mask1 = (0 <= refer_pts[0, :]) & (refer_pts[0, :] < self.size[0]) & (0 <= refer_pts[1, :]) & (
                    refer_pts[1, :] < self.size[1])
        mask = mask1  # (1600,)

        match_index = np.int32(refer_pts[0, :] // self.stride + (refer_pts[1, :] // self.stride) * ncols)
        match_index = np.clip(match_index, 0, ncols * nrows - 1)  # 确保 match_index 在范围内

        indexes = np.arange(nrows * ncols)[mask]
        indexes = np.clip(indexes, 0, ncols * nrows - 1)  # 确保 indexes 在范围内

        for index in indexes:
            label[index][match_index[index]] = 1

        return label

    def _make_query_pts(self):
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        half_stride = (self.stride-1) / 2
        xs = np.arange(ncols)
        ys = np.arange(nrows)
        xs = np.tile(xs[np.newaxis,:],(nrows,1))
        ys = np.tile(ys[:,np.newaxis],(1,ncols))
        ones = np.ones((nrows,ncols,1))
        grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis],ones],axis=-1)
        grid[:,:,:2] = grid[:,:,:2] * self.stride + half_stride  #(0:20, 0:20, 1) , shape:20x20x3
        return grid.reshape(-1,3) # (nrows*ncols , 3)

    def _random_flag(self,thresh=-1):
        return np.random.rand(1) < thresh

    def _random_crop(self, img):
        h, w, c = img.shape
        crop_height, crop_width = 320, 320  # 裁剪后的图像尺寸

        # 确保裁剪区域不超出图像边界
        x = random.randint(0, max(w - crop_width, 0))  # 随机生成 x 坐标
        y = random.randint(0, max(h - crop_height, 0))  # 随机生成 y 坐标

        # 裁剪图像
        img_cropped = img[y:y + crop_height, x:x + crop_width]

        # 构建裁剪变换矩阵
        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

        return img_cropped, crop_M, (x, y)

    def _random_crop3(self, img, x=None, y=None):
        h, w, c = img.shape
        crop_height, crop_width = 320, 320  # 裁剪后的图像尺寸

        # 固定裁剪位置
        x = 0  # 固定 x 坐标
        y = 0  # 固定 y 坐标

        # 确保裁剪区域不超出图像边界
        x = max(0, min(x, w - crop_width))  # 确保 x 在有效范围内
        y = max(0, min(y, h - crop_height))  # 确保 y 在有效范围内

        # 裁剪图像
        img_cropped = img[y:y + crop_height, x:x + crop_width]

        # 构建裁剪变换矩阵
        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

        return img_cropped, crop_M, (x, y)

    def _random_crop2(self, img):
        h, w, c = img.shape
        crop_height, crop_width = 320, 320  # 裁剪后的图像尺寸

        # 随机生成 x 和 y 坐标
        x = random.randint(100, 110)  # x 在 240 到 270 之间随机
        y = random.randint(80, 100)  # y 在 120 到 150 之间随机

        # 确保裁剪区域不超出图像边界
        x = max(0, min(x, w - crop_width))  # 确保 x 在有效范围内
        y = max(0, min(y, h - crop_height))  # 确保 y 在有效范围内

        # 裁剪图像
        img_cropped = img[y:y + crop_height, x:x + crop_width]

        # 构建裁剪变换矩阵
        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

        return img_cropped, crop_M, (x, y)

    def _aug_img(self,img, src, qc, aug=1):
        h,w = img.shape[:2]
        matrix = np.eye(3)
 
        if self._random_flag(aug):
            img,rM = random_rotation2(img,src, qc, max_degree=45)
            #img,rM = random_rotation(img, max_degree=45)
            rM = np.concatenate([rM,np.array([[0,0,1]],np.float32)])
            matrix = np.matmul(rM,matrix)

        if self._random_flag(aug * 0.5):
            kernel = random.choice([3,5,7])
            img = blur_image(img,kernel)

        if self._random_flag(aug * 0.5):
            img = img[:,::-1,...].copy() # horizontal flip
            fM = np.array([
                [-1,0,w-1],
                [0,1,0],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(fM,matrix)

        if self._random_flag(aug * 0.2):
            img = img[::-1,:,...].copy() # vertical flip
            vfM = np.array([
                [1,0,0],
                [0,-1,h-1],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(vfM,matrix)

        if self._random_flag():
            img = change_lightness_contrast(img) # change light

        if self._random_flag():
            h,s,v = np.random.rand(3)/2.5 - 0.2
            img = random_distort_hsv(img,h,s,v)

        if self._random_flag(aug * 0.1):
            img = random_gauss_noise(img)

        if self._random_flag():
            img = random_mask(img)

        if self._random_flag():
            img,sh,sw = random_jitter(img,max_jitter=0.3)
            jM = np.array([
                [1,0,sw],
                [0,1,sh],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(jM,matrix)


        return img,matrix

    def __len__(self):
        return len(self.train_data)


def build_Rotate_RGBD(
        train_data_file,
        test_data_file,
        size,
        stride):
    train_data = RotateRGBDDataset(
        train_data_file,
        size=(320, 320),
        stride=8,
        aug=True)
    test_data = RotateRGBDDataset(
        test_data_file,
        size=(320, 320),
        stride=8, 
        aug=False)

    return train_data, test_data


if __name__ == "__main__":
    from utils import _transform_inv,draw_match
    size = (320,320)
    dataloader = DataLoader(
        RotateRGBDDataset("D:/study/data/rgbd/train.txt",size=size,aug=True),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    print(len(dataloader))
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)
    mean_dep = np.array([0.30097574, 0.30097574, 0.30097574],dtype=np.float32).reshape(3,1,1)
    std_dep = np.array([0.08920097, 0.08920097, 0.08920097],dtype=np.float32).reshape(3,1,1)
    check_index = 0
    num = 0
    while 1:
        for sample in dataloader:
            query,refer,label_matrix = sample["query"],sample["refer"],sample["gt_matrix"]
            query0 = query.detach().cpu().numpy()[check_index]
            refer0 = refer.detach().cpu().numpy()[check_index]
            label_matrix0 = label_matrix.detach().cpu().numpy()[check_index]
            query1 = query.detach().cpu().numpy()[check_index+1]
            refer1 = refer.detach().cpu().numpy()[check_index+1]
            label_matrix1 = label_matrix.detach().cpu().numpy()[check_index+1]

            sq0 = _transform_inv(query0,mean_dep,std_dep)
            sr0 = _transform_inv(refer0,mean,std)
            heatmapshow = None
            heatmapshow = cv2.normalize(sq0, heatmapshow, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            out0 = draw_match(label_matrix0>0,heatmapshow, sr0).squeeze()
            sq1 = _transform_inv(query1,mean_dep,std_dep)
            sr1 = _transform_inv(refer1,mean,std)
            heatmapshow = None
            heatmapshow = cv2.normalize(sq1, heatmapshow, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            out1 = draw_match(label_matrix1>0, heatmapshow, sr1).squeeze()
            cv2.imwrite(f"images/match_img0{num}.jpg",out0)
            cv2.imwrite(f"images/match_img1{num}.jpg",out1)
            num = num+ 1

