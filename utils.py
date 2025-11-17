import numpy as np
from scipy import ndimage


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
    # 进行均值计算
    R_Mean = np.mean(R)
    G_Mean = np.mean(G)
    B_Mean = np.mean(B)

    K = R_Mean + G_Mean + B_Mean

    flag = False
    # 若绿色通道的均值占比超过2/3，则返回原图
    if G_Mean / K > 2 / 3:
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


def simple_color_balance(img):
    """
    简单颜色平衡
    """
    num = 255
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    # 找到最大平均值并计算比例
    max = np.max([r_mean, g_mean, b_mean])
    ratio = [max / r_mean, max / g_mean, max / b_mean]

    # 计算饱和度水平
    satLevel = 0.005 * np.array(ratio)
    # 获取图像尺寸
    m, n, p = img.shape

    # 重塑图像数据
    imgRGB_orig = np.zeros((p, m * n))
    for i in range(p):
        imgRGB_orig[i, :] = img[:, :, i].flatten()

    # 初始化输出数组
    imRGB = np.zeros(imgRGB_orig.shape)

    # 对每个通道进行处理
    for ch in range(p):
        # 计算分位数阈值
        q = [satLevel[ch], 1 - satLevel[ch]]
        tiles = np.quantile(imgRGB_orig[ch, :], q)

        # 截断超出范围的值
        temp = imgRGB_orig[ch, :].copy()
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        imRGB[ch, :] = temp

        # 归一化到0-255范围
        pmin = np.min(imRGB[ch, :])
        pmax = np.max(imRGB[ch, :])
        imRGB[ch, :] = (imRGB[ch, :] - pmin) * num / (pmax - pmin)

    # 重构图像
    output = np.zeros(img.shape)
    for i in range(p):
        output[:, :, i] = imRGB[i, :].reshape((m, n))

    # 转换为uint8格式
    output = output.astype(np.uint8)
    return output

def average_decomposition(I, alpha=1, hsize=31):
    """
    双尺度图像分解算法
    参数:
    I: 输入图像
    alpha: 增强因子
    hsize: 平均滤波器大小
    返回:
    result: 处理后的图像
    B: 低频分量
    D: 高频分量
    """
    # 将图像转换为double类型（0-1范围）
    I = I.astype(np.float64) / 255.0
    
    # 创建平均滤波器
    f = np.ones((hsize, hsize)) / (hsize * hsize)
    
    # 对每个通道分别应用平均滤波器获取低频分量
    if len(I.shape) == 3:  # 彩色图像
        B = np.zeros_like(I)
        for i in range(I.shape[2]):  # 对每个通道进行处理
            B[:, :, i] = ndimage.convolve(I[:, :, i], f, mode='nearest')
    else:  # 灰度图像
        B = ndimage.convolve(I, f, mode='nearest')
    
    # 计算高频分量
    D = I - B
    
    # 结合原始图像和增强的高频分量
    result = I + alpha * D
    
    return result, B, D