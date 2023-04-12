import torch
import torch.nn as nn
import numpy as np
from .rangenet import RangeNet
from .attention import ChannelAttention, SpatialAttention

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter_range_image(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        print('grid_size:, ', grid_size)
        assert self.nz == 1

        self.rangenet = RangeNet(self.model_cfg)
        self.num_pillar_features = 64

        self.ca = ChannelAttention(self.num_pillar_features)
        self.sa = SpatialAttention()


    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_pillar_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_pillar_features * self.nz, self.ny, self.nx)

        # range net
        range_in = batch_dict['laser_range_in'].permute(0, 3, 1, 2).contiguous()
        # range net forward
        down, up = self.rangenet(range_in)  # out 1*64*64*1024 N*C*H*W
        output = up[len(up)-1]
        
        output = self.ca(output) * output
        output = self.sa(output) * output
        
        batch_size = range_in.shape[0]
        # print('range_in shape,', range_in.size())
        x = batch_dict['laser_x']
        y = batch_dict['laser_y']
        ord_p = batch_dict['laser_points']
        # range_imgs = batch_dict['range_imgs']
        # print('range_imgs shape,', range_imgs.size())

        # print('x shape,', x.size())
        # print('y shape,', y.size())
        # print('ord_p shape,', ord_p.size())

        bevs = []
        for batch_idx in range(batch_size):
            x_batch = x[torch.where(x[:, 0] == batch_idx)][:, -1]
            y_batch = y[torch.where(y[:, 0] == batch_idx)][:, -1]
            ord_p_batch = ord_p[torch.where(ord_p[:, 0] == batch_idx)][:, 1:].permute(1, 0)
            # print('batch_idx,', batch_idx)
            # print('y_batch,', y_batch.size())
            # print('x_batch,', x_batch.size())
            output_batch = output[batch_idx, :, y_batch.long(), x_batch.long()]
            
            # print('ord_p_batch,', ord_p_batch.size())
            # print('output_batch,', output_batch.size())
            res = torch.cat((ord_p_batch, output_batch), 0)
            # print('res shape', res.size())
            
            bev = gen_bev_map(res) # kitti
            
            
            
            bevs.append(torch.unsqueeze(bev, 0))

        bevs_torch = torch.cat((bevs),0)
        # print('bev size',bevs_torch.size())
        # print('batch_spatial_features size',batch_spatial_features.size())
        batch_add = torch.cat((batch_spatial_features, bevs_torch), dim=1)
       
        batch_dict['spatial_features'] = batch_add



        return batch_dict


def gen_bev_map(pc, y_range=[-39.68, 39.68], x_range=[0, 69.12], res=0.16):
    # bc = pc.shape[0]
    c, n = pc.shape
    c = c - 4

    w = int((y_range[1] - y_range[0])/res)
    h = int((x_range[1] - x_range[0])/res)

    # for waymo
    w, h = 468, 468
    # print(w, h)

    point = pc.permute(1,0).contiguous()
    point = point.float().cpu().detach().numpy()

    x = point[:,0]
    y = point[:,1]
    z = point[:,2]

    # print('x', x.shape)
    # print('y', y.shape)
    # print('z', z.shape)

    im = np.zeros((h, w, c), dtype=np.float32)


    # filter point cloud
    f_filt = np.logical_and((x>x_range[0]), (x<x_range[1]))
    s_filt = np.logical_and((y>y_range[0]), (y<y_range[1]))
    filt = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filt).flatten()
    x = x[indices]
    y = y[indices]
    z = z[indices]
    point = point[:, 4:][indices]

    # convert coordinates to 
    x_img = (-y/res).astype(np.int32)
    y_img = (-x/res).astype(np.int32)
    # shifting image, make min pixel is 0,0
    x_img -= int(np.floor(y_range[0]/res))
    y_img += int(np.ceil(x_range[1]/res))

    x_max = int((y_range[1]-y_range[0])/res-1)
    y_max = int((x_range[1]-x_range[0])/res-1)

    x_img = np.clip(a=x_img, a_min=0, a_max=x_max)
    y_img = np.clip(a=y_img, a_min=0, a_max=y_max)

    x_filt = np.logical_and((x>=0), (x<h))
    y_filt = np.logical_and((y>=0), (y<w))
    filt = np.logical_and(x_filt, y_filt)
    indices = np.argwhere(filt).flatten()
    x_img = x_img[indices]
    y_img = y_img[indices]
    z = z[indices]
    point = point[indices]


    # crop z to make it not bigger than 255
    # height_range = (-3, 1)
    height_range = (-2, 4)
    
    z_c = np.clip(a=z, a_min=height_range[0], a_max=height_range[1])

    '''
    def scale_to_255(a, min, max, dtype=np.uint8):

        return (((a - min) / float(max - min)) * 255).astype(dtype)

    pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])
    '''

    # according to width and height generate image
    z_c = z_c.reshape(-1,1)
    # print('im shape', im.shape)
    # print('x_img max', np.max(x_img))
    # print('y_img max', np.max(y_img))
    # print('point shape', point.shape)
    im[x_img, y_img] = z_c*point
    
    # im = torch.from_numpy(im).permute(2,1,0).contiguous()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return torch.from_numpy(im).float().to(device).permute(2,1,0).contiguous()
