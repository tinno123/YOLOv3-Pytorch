import torch






def get_ignore(feature_shape, anchors, targets_denormalized, stride,num_anchors_perscale ,positive_mask,positive_threshold = 0.5):

    # ---------------------------#
    # 获取网格中心并映射到原图尺寸
    # ---------------------------#
    grid_x, grid_y = torch.meshgrid(torch.arange(feature_shape[0]), torch.arange(feature_shape[1]), indexing='xy')
    grid_x = grid_x.flatten().repeat(num_anchors_perscale)  # 每个网格对应3个Anchor
    grid_y = grid_y.flatten().repeat(num_anchors_perscale)
    origin_center = torch.stack([(grid_x + 0.5) * stride, (grid_y + 0.5) * stride], dim=-1)

    # ---------------------------#
    # 拼接Anchors
    # ---------------------------#
    anchors = anchors.clone()
    anchors = torch.cat([origin_center,anchors.repeat(feature_shape[0] * feature_shape[1], 1)], dim=-1)

    # ---------------------------#
    # 计算iou
    # ---------------------------#
    iou = calculate_iou(targets_denormalized[:,1:5].unsqueeze(1), anchors.unsqueeze(0))

    # ---------------------------#
    # 筛选忽略样本
    # ---------------------------#
    ignore_mask = iou > positive_threshold
    ignore_mask = torch.any(ignore_mask, dim=0)

    return ignore_mask & ~(positive_mask.bool())


def build_labels(image_size, targets, anchors,stride = [ 32, 16, 8]):
    """
    :param image_size: 模型输入图像的大小[w,h]
    :param targets: 所有的真实标签[N,5] class,cx,cy,w,h（已归一化到0-1）
    :param anchors : 所有尺度的Anchors,从大到小排列[9,2]（像素值）
    :param stride: 下采样步长，顺序[32,16,8]
    :return:  label : [class_id, cx_offset, cy_offset, w_log, h_log, 正负样本标记mask, 大小目标权重平衡] [10647,7]
             各维度说明：
             - class_id: 目标类别ID（正样本有效，负样本/忽略样本为0）
             - cx_offset/cy_offset: 目标中心在特征图网格内的偏移（0~1）
             - w_log/h_log: 目标宽高/对应Anchor宽高的对数（正样本有效）
             - 正负样本标记mask: 1=正样本，0=负样本
    """
    # -----------------------------#
    # 获取先验框
    # -----------------------------#
    anchors = anchors

    # -----------------------------#
    # 获取下采样步长
    # -----------------------------#
    stride = stride
    if anchors.shape[0] % len(stride) !=0 :
        raise ValueError('特征图尺度数量与先验框数量不匹配，请重新配置')

    # -----------------------------#
    # 获取每一个尺度的先验框数量
    # -----------------------------#
    num_anchors_perscale = anchors.shape[0] // len(stride)

    # -----------------------------#
    # 获取每一个尺度的特征图大小
    # -----------------------------#
    feature_shape =[ [image_size[0]//i,image_size[1]//i]  for i in stride]

    # -----------------------------#
    # 获取反归一化和归一化标签
    # -----------------------------#
    targets_denormalized = targets.clone()
    targets_normalized = targets.clone()
    anchors_normalized = anchors.clone().float()
    # 反归一化真实标注
    targets_denormalized[:, [1,3]] = targets_normalized[:, [1,3]] * image_size[0]
    targets_denormalized[:, [2,4]] = targets_normalized[:, [2,4]] * image_size[1]
    # 归一化Anchors
    anchors_normalized[:, 0] = anchors_normalized[:, 0] / image_size[0]
    anchors_normalized[:, 1] = anchors_normalized[:, 1] / image_size[1]






    # -----------------------------#
    # 为每一个尺度的特征图生成标签
    # -----------------------------#
    results = []
    for i in range(len(stride)):
        targets_wh = targets_denormalized[:, [3, 4]]
        anchors_wh = anchors
        center_xy = torch.tensor([0,0]).unsqueeze(0)
        result = torch.zeros([feature_shape[i][0] *  feature_shape[i][1] * num_anchors_perscale, 11])

        #----------------------------#
        # 构造同一个中心坐标
        # ---------------------------#
        targets_xywh = torch.cat([center_xy.repeat(targets_wh.shape[0],1),targets_wh], dim=-1)
        anchors_xywh = torch.cat([center_xy.repeat(anchors_wh.shape[0],1),anchors_wh], dim=-1)

        # -------------------------#
        # 计算iou
        # -------------------------#
        iou = calculate_iou(targets_xywh.unsqueeze(1), anchors_xywh.unsqueeze(0))

        # -------------------------#
        # 判断哪些属于当前尺度
        # -------------------------#
        max_iou_index = torch.argmax(iou, dim=1)
        scale_index = max_iou_index  // ( anchors.shape[0] // len(stride))
        current_fit_index = torch.where(scale_index == i)[0]
        if len(current_fit_index) == 0 :
            results.append(result)
            continue

        # -------------------------#
        # 映射到特征图
        # -------------------------#
        current_fit_target = targets_normalized[current_fit_index]
        current_fit_target[:, 1] =current_fit_target[:, 1] * feature_shape[i][0]
        current_fit_target[:, 2] =current_fit_target[:, 2] * feature_shape[i][1]

        # -------------------------#
        # 构造标签数据
        # -------------------------#
        labels_xy = current_fit_target[:, [1, 2]] - torch.floor(current_fit_target[:, [1, 2]])
        labels_wh = torch.log(torch.clamp(current_fit_target[:, [3, 4]] / anchors_normalized[max_iou_index[current_fit_index]], min=1e-6))
        labels_id = current_fit_target[:,0]

        # -------------------------#
        # 计算插入位置
        # -------------------------#
        xmin = torch.floor(current_fit_target[:, 1])
        ymin = torch.floor(current_fit_target[:, 2])
        insert_index = ( (ymin * feature_shape[i][0] + xmin) * num_anchors_perscale  + max_iou_index[current_fit_index] % num_anchors_perscale ).int().unsqueeze(1)
        # print("匹配的坐标",torch.stack([xmin,ymin],dim=-1))
        # print("匹配的Anchor",max_iou_index[current_fit_index] % num_anchors_perscale)
        # print("类别标签",current_fit_target[:,0])
        # print()
        # -------------------------#
        # 计算正样本mask
        # -------------------------#
        try:
            result[insert_index, 5] = 1
        except Exception as e:
            print(ymin)
            print(xmin)

        # -------------------------#
        # 插入标签数据
        # -------------------------#
        result[insert_index,0] = labels_id.unsqueeze(-1)
        result[insert_index, [1,2]] = labels_xy
        result[insert_index, [3,4]] = labels_wh
        result[insert_index,  [6,7,8,9]] =targets_normalized[current_fit_index][:,1:]
        result[insert_index,-1] = 2 - (targets_normalized[current_fit_index][:,3] * targets_normalized[current_fit_index][:,4]).unsqueeze(-1)
        results.append(result)

    results = torch.cat(results, dim=0)
    return results



if __name__ == '__main__':
    targets = torch.tensor([
        [0, 300 / 416, 300 / 416, 373 / 416, 236 / 416],
        [1, 200 / 416, 100 / 416, 156 / 416, 198 / 416],
        [0, 34 / 416, 300 / 416, 62 / 416, 45 / 416],
        [1, 90 / 416, 100 / 416, 33 / 416, 23 / 416],
    ])
    size = torch.tensor([[13, 13], [26, 26], [52, 52]])  # [3,2]代表3个特征图的形状
    anchors_config = torch.tensor([
        [116, 90] ,[156, 198],[373, 326],
        [30, 61],[62, 45], [59, 119],
        [10, 13], [16, 30], [33, 23]
    ])

    result = build_labels([416, 416], targets, anchors_config)




