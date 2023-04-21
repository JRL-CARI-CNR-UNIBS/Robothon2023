#!/usr/bin/env python3

from PIL import Image, ImageFilter, ImageDraw
import os
import cv2
import numpy


path = '/home/gauss/projects/personal_ws/src/tests_package/foto_board'
dir_list = os.listdir(path)

#for image_n in range(500):
wo = True
while (wo):
    image_n = 999
    dir_list = {}
    dir_list[image_n] = '999'
#    print(image_n)
#    try:
#        img  = Image.open('/home/gauss/projects/personal_ws/src/tests_package/foto_board/' + dir_list[image_n])
#    except IOError:
#        pass
    img  = Image.open('/home/gauss/projects/personal_ws/src/tests_package/foto_board/screen_230420_134418.png')

    image_name          = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '.jpg'
    crop_image_name     = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '_crop.jpg'
    bw_image_name       = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '_bw.jpg'
    clear_image_name    = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '_clear.jpg'
    edge_image_name     = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '_edge.jpg'
    edge_image_name_c   = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '_edge_c.jpg'
    edge_image_name_c_c = '/home/gauss/projects/personal_ws/src/tests_package/foto_board_/Image_' + str(image_n) + '_edge_c_c.jpg'

    rgb_im = img.convert('RGB')
    width, height = img.size

    max_x = 0
    min_x = width
    max_y = 0
    min_y = height

    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))

            if ( r < 20 ):
                if ( 200 < g):
                    if ( 200 < b):
                        if ( x > max_x ):
                            max_x = x
                        if ( x < min_x ):
                            min_x = x
                        if ( y > max_y ):
                            max_y = y
                        if ( y < min_y ):
                            min_y = y

    if ((max_x == 0) or (max_y == 0) or (min_x == width) or (min_y == height)):
        print('Image ' + dir_list[image_n] + ' have not azzure screen')
        img.save(image_name)
        continue

    area = (min_x+2, min_y, max_x, max_y)
    ref_min_x1 = min_x
    ref_min_y1 = min_y
    ref_max_x1 = max_x
    ref_max_y1 = max_y
    img_c = img.crop(area)

    img_bw = img_c.copy()
    width, height = img_bw.size
    img_clean = img_c.copy()

    for x in range(width):
        for y in range(height):
            r, g, b = img_c.getpixel((x, y))

            if ( 180 < r ):
                if ( 235 < g):
                    if ( 235 < b):
                        img_bw.putpixel((x,y), (255,255,255))
                    else:
                        img_bw.putpixel((x,y), (0,0,0))
                else:
                    img_bw.putpixel((x,y), (0,0,0))
            else:
                img_bw.putpixel((x,y), (0,0,0))
            if ( r < 190 ):
                img_clean.putpixel((x,y), (0,0,0))
            else:
                img_clean.putpixel((x,y), (255,255,255))
#    img_clean.show()

    cv_img = numpy.array(img_clean)


    t_lower = 100  # Lower Threshold
    t_upper = 250  # Upper threshold

    cv_img = cv2.GaussianBlur(cv_img, (3,3), 0)

    # Applying the Canny Edge filter
    edge = cv2.Canny(cv_img, t_lower, t_upper)

    img_final = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_final)
    im_pil_ = im_pil.copy()

    width, height = img_bw.size
    finish = False

    for y in range(height):
        for x in range(width):
            r, g, b = img_bw.getpixel((x, y))
            if ( r > 100 ):
                h_min = y - 2
                h_max = y + 50
                finish = True
                break
        if (finish):
            break

    area = (0, h_min, width, h_max)
    ref_min_x2 = 0
    ref_min_y2 = h_min
    ref_max_x2 = width
    ref_max_y2 = h_max

    im_pil = im_pil.crop(area)
    im_pil__ = im_pil.copy()
    img_bw = img_bw.crop(area)

    width, height = im_pil.size
    finish = False

    for y in range(height-1, 0, -1):
        for x in range(width):
            r, g, b = im_pil.getpixel((x, y))
            if ( r > 100 ):
                h_max = y + 2
                finish = True
                break
        if (finish):
            break

    area = (0, 0, width, h_max)
    print(width)
    print(width/2)
    ref_min_x3 = 0
    ref_min_y3 = 0
    ref_max_x3 = width
    ref_max_y3 = h_max

    im_pil = im_pil.crop(area)

    y1 = 5
    y2 = h_max - 7

    up_point = []
    down_point = []
    r1_old = 0
    g1_old = 0
    b1_old = 0
    r2_old = 0
    g2_old = 0
    b2_old = 0

    for x in range(width):
        r, g, b = im_pil.getpixel((x, y1))
        if ( r - r1_old > 100 ):
            up_point.append(x)
        r, g, b = im_pil.getpixel((x, y2))
        if ( r - r2_old > 100 ):
            down_point.append(x)
        r1_old, g1_old, b1_old = im_pil.getpixel((x, y1))
        r2_old, g2_old, b2_old = im_pil.getpixel((x, y2))

    point_x = []
    point_y = []

    if ( (y2-y1) > 20 ):
        if ( len(up_point) == 2 ):
            if ( (up_point[1] - up_point[0]) > 22 ):
                x1 = up_point[0] + 10
                x2 = up_point[1] - 10
            else:
                x1 = round((up_point[1] + up_point[0]) / 2)
                x2 = round((up_point[1] + up_point[0]) / 2)
            im_pil.putpixel((x1,y1), (255,0,0))
            im_pil.putpixel((x1+1,y1), (255,0,0))
            im_pil.putpixel((x1-1,y1), (255,0,0))
            im_pil.putpixel((x1,y1+1), (255,0,0))
            im_pil.putpixel((x1,y1-1), (255,0,0))
            im_pil.putpixel((x2,y1), (255,0,0))
            im_pil.putpixel((x2+1,y1), (255,0,0))
            im_pil.putpixel((x2-1,y1), (255,0,0))
            im_pil.putpixel((x2,y1+1), (255,0,0))
            im_pil.putpixel((x2,y1-1), (255,0,0))
            point_x.append(x1)
            point_y.append(y1)
            point_x.append(x2)
            point_y.append(y1)
        elif ( len(up_point) == 3 ):
            x1 = up_point[0] + 10
            x2 = up_point[2] - 10
            im_pil.putpixel((x1,y1), (255,0,0))
            im_pil.putpixel((x1+1,y1), (255,0,0))
            im_pil.putpixel((x1-1,y1), (255,0,0))
            im_pil.putpixel((x1,y1+1), (255,0,0))
            im_pil.putpixel((x1,y1-1), (255,0,0))
            im_pil.putpixel((x2,y1), (255,0,0))
            im_pil.putpixel((x2+1,y1), (255,0,0))
            im_pil.putpixel((x2-1,y1), (255,0,0))
            im_pil.putpixel((x2,y1+1), (255,0,0))
            im_pil.putpixel((x2,y1-1), (255,0,0))
            point_x.append(x1)
            point_y.append(y1)
            point_x.append(x2)
            point_y.append(y1)
        elif ( (len(up_point) == 4) ):
            x1 = round((up_point[1] + up_point[0]) / 2)
            x2 = round((up_point[3] + up_point[2]) / 2)
            im_pil.putpixel((x1,y1), (255,0,0))
            im_pil.putpixel((x1+1,y1), (255,0,0))
            im_pil.putpixel((x1-1,y1), (255,0,0))
            im_pil.putpixel((x1,y1+1), (255,0,0))
            im_pil.putpixel((x1,y1-1), (255,0,0))
            im_pil.putpixel((x2,y1), (255,0,0))
            im_pil.putpixel((x2+1,y1), (255,0,0))
            im_pil.putpixel((x2-1,y1), (255,0,0))
            im_pil.putpixel((x2,y1+1), (255,0,0))
            im_pil.putpixel((x2,y1-1), (255,0,0))
            point_x.append(x1)
            point_y.append(y1)
            point_x.append(x2)
            point_y.append(y1)
        else:
            print('Image ' + dir_list[image_n] + ' error with up triangles: ' + str(len(up_point)) )
            im_pil.save(edge_image_name_c_c)
            img.save(image_name)
            continue
        if ( len(down_point) == 1 ):
            if ( down_point[0] < 80 ):
                x = down_point[0] - 10
                if ( x < 0 ):
                    x = 0
            else:
                x = down_point[0] + 10
                if ( x >= width ):
                    x = width - 1
            im_pil.putpixel((x,y2), (255,0,0))
            im_pil.putpixel((x+1,y2), (255,0,0))
            im_pil.putpixel((x-1,y2), (255,0,0))
            im_pil.putpixel((x,y2+1), (255,0,0))
            im_pil.putpixel((x,y2-1), (255,0,0))
            point_x.append(x)
            point_y.append(y2)
        elif ( len(down_point) == 2 ):
            if ( (down_point[1] - down_point[0]) > 22 ):
                print('Image ' + dir_list[image_n] + ' error with down triangle: ' + str(len(down_point)) )
                continue
            x = round((down_point[1] + down_point[0]) / 2)
            im_pil.putpixel((x,y2), (255,0,0))
            im_pil.putpixel((x+1,y2), (255,0,0))
            im_pil.putpixel((x-1,y2), (255,0,0))
            im_pil.putpixel((x,y2+1), (255,0,0))
            im_pil.putpixel((x,y2-1), (255,0,0))
            point_x.append(x)
            point_y.append(y2)
        else:
            print('Image ' + dir_list[image_n] + ' error with down triangle: ' + str(len(down_point)) )
            im_pil.save(edge_image_name_c_c)
            img.save(image_name)
            continue

    else:
        print('Image ' + dir_list[image_n] + ' not found all triangles')
        img.save(image_name)
        img_c.save(crop_image_name)
        img_bw.save(bw_image_name)
        img_clean.save(clear_image_name)
        im_pil.save(edge_image_name_c_c)
        im_pil_.save(edge_image_name)
        im_pil__.save(edge_image_name_c)
        img.save(image_name)
        continue
    finalx = [] # posizione assoluta nell'immagine
    finaly = []
# point_x posiziome all'interno del crop
    print('Point_c: x: ' + str(point_x))
    print('         y: ' + str(point_y))

    for i in range(len(point_x)):
        finalx.append( point_x[i] + ref_min_x1 + ref_min_x2 + ref_min_x3 )
        finaly.append( point_y[i] + ref_min_y1 + ref_min_y2 + ref_min_y3 )
    for i in range(len(finalx)):
        if ( 85 < point_x[i] < 95 ):
            img.putpixel((finalx[i],     finaly[i]),     (0,255,0))
            img.putpixel((finalx[i] + 1, finaly[i]),     (0,255,0))
            img.putpixel((finalx[i] - 1, finaly[i]),     (0,255,0))
            img.putpixel((finalx[i],     finaly[i] + 1), (0,255,0))
            img.putpixel((finalx[i],     finaly[i] - 1), (0,255,0))
        else:
            img.putpixel((finalx[i],     finaly[i]),     (255,0,0))
            img.putpixel((finalx[i] + 1, finaly[i]),     (255,0,0))
            img.putpixel((finalx[i] - 1, finaly[i]),     (255,0,0))
            img.putpixel((finalx[i],     finaly[i] + 1), (255,0,0))
            img.putpixel((finalx[i],     finaly[i] - 1), (255,0,0))
    if ( len(finalx) == 0):
        img.save(image_name)
        img_c.save(crop_image_name)
        img_bw.save(bw_image_name)
        img_clean.save(clear_image_name)
        im_pil.save(edge_image_name_c_c)
        im_pil_.save(edge_image_name)
        im_pil__.save(edge_image_name_c)
        img.save(image_name)
        continue
    print('Point: x: ' + str(finalx))
    print('       y: ' + str(finaly))
    im_pil.save(edge_image_name_c_c)
    img.save(image_name)

    wo = False
