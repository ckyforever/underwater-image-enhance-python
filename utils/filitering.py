import numpy as np


def compensate3(img):
    """
    自适应颜色校正
    """
    img = img.astype(np.float32) / 255.0
    ret = img.copy()
    # 获取各个通道到的值
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    #进行均值计算
    R_Mean = np.mean(R)
    G_Mean = np.mean(G)
    B_Mean = np.mean(B)

    K = R_Mean + G_Mean + B_Mean 

    flag = False
    #若绿色通道的均值占比超过2/3，则返回原图
    if G_Mean / K > 2 /3 :
        ret = (img * 255).astype(np.uint8)
        return ret
    
    # 计算alpha，用于红色通道的调整
    alpha = 1 - np.log(G_Mean - R_Mean)

    # 绿色通道与蓝色通道差异较大时的处理
    if G_Mean - B_Mean >= 0.1:
        # 计算beta参数
        beta = -np.log(G_Mean - B_Mean)
        
        # 根据红蓝通道差异调整beta和alpha
        if R_Mean - B_Mean <= 0.1:
            beta = 1 / (1 + beta)
        elif R_Mean - B_Mean > 0.1:
            beta = 1 + beta
            alpha = 0
        
        # 调整蓝色通道值
        B = B + (G_Mean - B_Mean) * (1 - B) * G * beta
    
    # 蓝绿通道差异较大时的处理
    if B_Mean - G_Mean > 0.1:
        # 计算gamma参数
        gamma = -np.log(B_Mean - R_Mean)
        # 调整绿色通道值
        G = G + (B_Mean - R_Mean) * (1 - G) * B * gamma
    
    # 调整红色通道值
    R = R + (G_Mean - R_Mean) * (1 - R) * G * alpha
    
    # 合并三个通道并转换为uint8格式
    ret = np.stack([R, G, B], axis=2)
    ret = (ret * 255).astype(np.uint8)
    return ret


