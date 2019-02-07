import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import transforms
import tensorflow_tut as tut
import tensorflow as tf

fig, ax = plt.subplots()
img_path = 'sudoku_sample_2.jpg'
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
print("CONTOUR EXTREME POINTS: ", points)
cropp = transforms.crop_square(transformed, 5, 0)

digits = []

# for i in range(9):
#   for j in range(9):
#     d = transforms.crop_square(transformed, i, j)
#     digits.append(d)

# plt.set_cmap(cmap)

# for i in range(10):
#   plt.imshow(digits[i])
#   plt.show()

# model = tut.create_model()
# predictions = model.predict([one_digit])
# print("PREDICITNG IT IS: ", np.argmax(predictions))

# for i in range(9):
#   for j in range(9):
#     digits.append(transforms.crop_square(transformed, i, j))

# for digit in digits:
#   a,b = digit.shape

# plt.subplot(221)
plt.imshow(sudoku_img)
plt.title('sudoku image: ')
plt.axis('off')
# plt.set_cmap(cmap)
plt.show()
# plt.subplot(222)
plt.imshow(transformed)
plt.title('biggest contour: ')
plt.axis('off')
# plt.set_cmap(cmap)
plt.show()

# plt.subplot(421)
# plt.imshow(one_digit)
# plt.title('digit: ')
# plt.axis('off')
# plt.set_cmap(cmap)

# plt.subplot(422)
# plt.imshow(cropp)
# plt.title('cropp: ')
# plt.axis('off')
# plt.set_cmap(cmap)
# plt.show()
