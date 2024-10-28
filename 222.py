import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize_image(image, size=(420, 344)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def high_pass_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 15
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    # cv2.imshow('Resized Image', mask[:,:,0]*255)
    # cv2.imshow('Resized Image2', mask[:,:,1]*255)
    # cv2.imshow('Resized Image3', dft_shift)
    fshift = dft_shift * mask.astype(np.float32)
    # fshift = dft_shift
    # cv2.imshow('Resized Image4', fshift)
    # cv2.waitKey(0)
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
def equalize(image):
    min, max = image.min(), image.max()
    result = (image - min) / (max - min)
    return result

def main():
    # 读取原图像
    image = cv2.imread('4.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # 调整图像大小到420x344
    resized_image = resize_image(image, size=(420, 344))

    # 高通滤波
    high_pass_result1 = high_pass_filter(resized_image)
    high_pass_result1 = equalize(high_pass_result1)
    # min, max = high_pass_result1.min(), high_pass_result1.max()
    # high_pass_result = (high_pass_result1 - min) / (max - max)

    # 低通滤波
    low_pass_result1 = low_pass_filter(high_pass_result1)
    low_pass_result1 = equalize(low_pass_result1)

    # 低通滤波
    low_pass_result2 = low_pass_filter(resized_image)
    low_pass_result2 = equalize(low_pass_result2)
    # 高通滤波
    high_pass_result2 = high_pass_filter(low_pass_result2)
    high_pass_result2 = equalize(high_pass_result2)

    # 展示图像
    plt.figure(figsize=(42, 34))

    # plt.subplot(2, 2, 1)
    # plt.imshow(resized_image, cmap='gray')
    # plt.title('original image')
    # plt.axis('off')

    plt.subplot(2, 2, 1)
    plt.imshow(high_pass_result1, cmap='gray')
    plt.title('HLhigh pass filter')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(low_pass_result1, cmap='gray')
    plt.title('HLlow pass filter')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(high_pass_result2, cmap='gray')
    plt.title('LHhigh pass filter')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(low_pass_result2, cmap='gray')
    plt.title('LHlow pass filter')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
