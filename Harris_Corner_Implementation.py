import cv2
import matplotlib.pyplot as plt
import numpy as np

# reading the image => bgr format
# image_1 = cv2.imread("Dataset\\chessboard.jpg")
image_1 = cv2.imread("Dataset\Image4.jpg")

# make a copy of the original image
original = image_1.copy()

# haris corner detection in opencv works only with grayscale
# images.  harris does not work with color images

# convert to grayscale
image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

# convert the image to float32 (required format for calculatoin)
image_1_32 = np.float32(image_1_gray)

# apply harris corner detection
dst = cv2.cornerHarris(image_1_32, blockSize=2, ksize=3, k = 0.04)
# window size = 2 increase => fewer corners
#                 decrease => more corner detected
# edge detection size = 3 => increase - smooth gradients, less noise, miss fine details
#                      descrease - captures fine edges, more sensitive to noise
# sensitivity parameter =0.04 => control how strict the corner detection is
    # increase : more strict few corner
    # decrease: less strict, more corners including weak corners as well

# dilate teh results to mark the corners clearly
dst_dilated = cv2.dilate(dst, None)

# threshold for making corners
image_1[dst_dilated>0.02*dst_dilated.max()] = [0, 0, 255] #red color in BGR , [0, 255, 0]
# dst_dilated => contain the result from harris corner
# dst_dilated.max() - find the strongest corner responses
# increase 0.02 - max is increased - fewer points will be detected


# visulaization
plt.figure(figsize=(15, 4))

# original image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Orignal Image")
plt.axis("off")

# visualizing gray scale image
plt.subplot(1, 4, 2)
plt.imshow(image_1_gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# visualizing  harris corner => dst
plt.subplot(1, 4, 3)
plt.imshow(dst, cmap="gray")
plt.title("Harris Response")
plt.axis("off")

# visulizing of the final corners
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
plt.title("Corners detected")
plt.axis("off")

plt.tight_layout()
plt.show()
