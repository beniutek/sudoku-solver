import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import transforms

fig, ax = plt.subplots()
img_path = 'sudoku_sample.jpg'
cmap = plt.cm.gray
orig_img = cv2.imread(img_path)
binary_img = transforms.prepare_image(img_path)
sudoku_img = transforms.find_sudoku(binary_img)
biggest_contour = transforms.get_biggest_contour(sudoku_img)
print("got biggest contour on sudoku img")

x, y = sudoku_img.shape
img_extreme_points = np.float32([[x, y], [0, y], [x, 0], [0, 0]])

points = transforms.get_contour_extreme_points(sudoku_img, biggest_contour)
M = cv2.getPerspectiveTransform(points, img_extreme_points)
transformed = cv2.warpPerspective(sudoku_img, M, (x, y))
print("size x:y ", transformed.shape)

plt.subplot(231)
plt.imshow(orig_img)
plt.title('original image: ')
plt.axis('off')
plt.set_cmap(cmap)

plt.subplot(232)
plt.imshow(transformed)
plt.title('find sudoku: ')
plt.axis('off')
plt.set_cmap(cmap)
plt.show()

cropp = transforms.crop_square(transformed, 5, 0)

digits = []

for i in range(9):
  for j in range(9):
    digits.append(transforms.crop_square(transformed, i, j))


plt.subplot(231)
plt.imshow(cropp)
plt.title('cropp: ')
plt.axis('off')
plt.set_cmap(cmap)
plt.show()
