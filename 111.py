import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize_image(image, size=(344, 420)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def high_pass_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 25
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back

def low_pass_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 25
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back

def main():
    # 读取原图像
    image = cv2.imread('4.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # 调整图像大小到420x344
    resized_image = resize_image(image, size=(344, 420))

    # 高通滤波
    high_pass_result1 = high_pass_filter(resized_image)

    # 低通滤波
    low_pass_result1 = low_pass_filter(high_pass_result1)

    # 低通滤波
    low_pass_result2 = low_pass_filter(resized_image)

    # 高通滤波
    high_pass_result2 = high_pass_filter(low_pass_result2)

    cv2.imwrite('resized_image.png', resized_image)
    cv2.imwrite('HLhigh_pass_result.png', high_pass_result1)
    cv2.imwrite('HLlow_pass_result.png', low_pass_result1)
    cv2.imwrite('LHhigh_pass_result.png', high_pass_result2)
    cv2.imwrite('LHlow_pass_result.png', low_pass_result2)
    # 显示处理后的图像
    cv2.imshow('Resized Image', resized_image)
    cv2.imshow('High-Low High Pass Result', high_pass_result1)
    cv2.imshow('High-Low Low Pass Result', low_pass_result1)
    cv2.imshow('Low-High Low Pass Result', low_pass_result2)
    cv2.imshow('Low-High High Pass Result', high_pass_result2)

    # 等待用户按键
    cv2.waitKey(0)

    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
