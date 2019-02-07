import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

def prepare_image(img_path):
  # get grayscale image
  gray_img = cv2.imread(img_path, 0)
  bitwised_img = cv2.bitwise_not(gray_img)
  denoised_img = cv2.fastNlMeansDenoising(bitwised_img, None, 10, 7, 21)
  thresholded_img = cv2.adaptiveThreshold(denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)

  return thresholded_img

def find_sudoku(prepared_img):
  max_contour = get_biggest_contour(prepared_img)
  return cv2.drawContours(prepared_img, [max_contour], 0, (100, 40, 150), 10)
  # return crop_image(prepared_img, max_contour)

def get_biggest_contour(img):
  contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  return max(contours, key=cv2.contourArea)

def get_contour_extreme_points(img, contour):
  m_point = image_center(img)
  l1, l2, l3, l4 = 0, 0, 0, 0
  p1, p2, p3, p4 = 0, 0, 0, 0

  for point in contour:
    d = distance(m_point, point[0])
    if inside_bottom_right(m_point, point[0]) and l1 < d:
      l1 = d
      p1 = point[0]
      continue
    if inside_bottom_left(m_point, point[0]) and l2 < d:
      l2 = d
      p2 = point[0]
      continue
    if inside_top_right(m_point, point[0]) and l3 < d:
      l3 = d
      p3 = point[0]
      continue
    if inside_top_left(m_point, point[0]) and l4 < d:
      l4 = d
      p4 = point[0]
      continue

  return np.float32([p1, p2, p3, p4])

def inside_bottom_right(center, point):
  return center[0] < point[0] and center[1] < point[1]

def inside_bottom_left(center, point):
  return center[0] > point[0] and center[1] < point[1]

def inside_top_right(center, point):
  return center[0] < point[0] and center[1] > point[1]

def inside_top_left(center, point):
  return center[0] > point[0] and center[1] > point[1]

def distance(p1, p2):
  return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

def image_center(img):
  x, y = img.shape

  return tuple([x/2, y/2])

def square_size(img):
  x, y = img.shape

  return (x / 9, y / 9)

def crop_square(img, i, j):
  xl, yl = square_size(img)

  return img[int(i*xl):int((i+1)*xl) ,int(j*yl):int((j+1)*yl)]
