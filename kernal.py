import cv2
import numpy as np

def apply_filter(image, kernel):
    # 对图像应用滤波器
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def main():
    # 读取图像
    image = cv2.imread('lenna_8.bmp')

    # 定义自定义的滤波器核
    # 这里只是一个示例，你可以根据需要自定义滤波器核
    custom_kernel = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
    # custom_kernel = np.array([[1, 0, -1],
    #                           [0, 0, 0],
    #                           [-1, 0, 1]])

    # 对图像应用自定义滤波器
    filtered_image = apply_filter(image, custom_kernel)

    # 显示原始图像和滤波后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
