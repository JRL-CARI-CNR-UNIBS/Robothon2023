#! /usr/bin/env python3

import cv2 as cv2


img = cv2.imread("frame_0.png")
cv2.imshow("test",img)
print(img)
cv2.waitKey(-1) 
cv2.destroyAllWindows()   