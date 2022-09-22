import cv2
import numpy

if __name__ == '__main__':
    # 初始化
    img = cv2.imread('image/2.1.png')
    img = cv2.resize(img, None, fx=3, fy=3)
    cv2.imshow("img", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col = gray.shape

    big_image = numpy.zeros()

    cv2.waitKey(0)
