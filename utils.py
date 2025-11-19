import numpy as np
from scipy import ndimage
import cv2
from skimage.color import rgb2hsv, hsv2rgb


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


def average_decomposition(img, alpha=1, hsize=31):
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
    img = img.astype(np.float64) / 255.0

    # 创建平均滤波器
    f = np.ones((hsize, hsize)) / (hsize * hsize)

    # 对每个通道分别应用平均滤波器获取低频分量
    if len(img.shape) == 3:  # 彩色图像
        B = np.zeros_like(img)
        for i in range(img.shape[2]):  # 对每个通道进行处理
            B[:, :, i] = ndimage.convolve(img[:, :, i], f, mode="nearest")
    else:  # 灰度图像
        B = ndimage.convolve(img, f, mode="nearest")

    # 计算高频分量
    D = img - B

    # 结合原始图像和增强的高频分量
    result = img + alpha * D

    return result, B, D


def gamma_correction_auto_2d(img):
    """
    2D自适应伽马校正函数
    该函数通过多尺度高斯滤波来自动确定局部伽马值，实现图像增强

    参数:
    input1: 输入图像（可以是彩色或灰度图像）

    返回:
    result: 经过伽马校正的图像
    """

    # 将输入图像转换为双精度浮点型（0-1范围）
    if img.dtype != np.float64:
        img = img.astype(np.float64) / 255.0

    # 检查是否为彩色图像（3通道）
    if len(img.shape) == 3:
        # 将RGB图像转换为HSV色彩空间
        HSV = rgb2hsv(img)
        V = HSV[:, :, 2]  # 提取明度通道

        # 获取图像尺寸
        height, width = img.shape[:2]
        filter_size = min(height, width)  # 确定滤波器大小

        # 设置多尺度高斯滤波参数
        c = [15, 80, 250]  # 三个不同的标准差参数
        q = np.sqrt(2)  # 缩放因子

        # OpenCV 的 ksize 必须是正奇数，sigmaX 控制模糊程度
        def blur_with_sigma(image, sigma):
            # 根据 sigma 自动计算合适的 kernel size（经验公式）
            ksize = int(2 * np.ceil(3 * sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            return cv2.GaussianBlur(
                image, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE
            )

        g1 = blur_with_sigma(V, c[0] / q)
        g2 = blur_with_sigma(V, c[1] / q)
        g3 = blur_with_sigma(V, c[2] / q)

        # 如果需要转回 float [0,1] 范围（与 skimage 一致）
        g1 = g1.astype(np.float64) / 255.0
        g2 = g2.astype(np.float64) / 255.0
        g3 = g3.astype(np.float64) / 255.0

        # 计算加权平均响应
        I = (g1 + g2 + g3) / 3

        # 计算全局平均亮度
        m = np.mean(I)

        # 根据局部亮度计算自适应伽马值
        the_gamma = np.power(0.5, (m - I) / m)

        # 应用伽马校正到明度通道
        HSV[:, :, 2] = np.power(V, the_gamma)

        # 将HSV转换回RGB色彩空间
        result = hsv2rgb(HSV)
    else:
        # 处理灰度图像
        V = img
        height, width = img.shape
        filter_size = min(height, width)

        # 设置多尺度高斯滤波参数
        c = [15, 80, 250]
        q = np.sqrt(2)

        # 创建三种不同尺度的高斯滤波器
        f1 = gaussian_filter(filter_size, c[0] / q)
        f2 = gaussian_filter(filter_size, c[1] / q)
        f3 = gaussian_filter(filter_size, c[2] / q)

        # 分别应用三个滤波器
        g1 = ndimage.convolve(V, f1, mode="reflect")
        g2 = ndimage.convolve(V, f2, mode="reflect")
        g3 = ndimage.convolve(V, f3, mode="reflect")

        # 计算加权平均响应
        I = (g1 + g2 + g3) / 3

        # 计算全局平均亮度
        m = np.mean(I)

        # 根据局部亮度计算自适应伽马值
        the_gamma = np.power(0.5, (m - I) / m)

        # 应用伽马校正
        V = np.power(V, the_gamma)
        result = V

    return (result * 255).astype(np.uint8)


def gaussian_filter(size, sigma):
    """
    创建高斯滤波器

    参数:
    size: 滤波器大小
    sigma: 高斯分布的标准差

    返回:
    二维高斯核
    """

    # 创建坐标网格
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)

    # 计算高斯核
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # 归一化
    return kernel / np.sum(kernel)
