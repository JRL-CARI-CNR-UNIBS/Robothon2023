#!/usr/bin/env python3

from PIL import Image, ImageFilter, ImageDraw
import os
import cv2
import numpy

def identification(img, red, yellow, green, path):

    width, height = img.size

    max_x = 0
    min_x = width
    max_y = 0
    min_y = height

#    screen serach
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))

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
        print('Image   have not azzure screen')
        print('Image   have not azzure screen')
        return False

    area = (min_x+2, min_y, max_x, max_y)
    ref_min_x1 = min_x
    ref_min_y1 = min_y
    ref_max_x1 = max_x
    ref_max_y1 = max_y
    img_c = img.crop(area)
    img_crop = img_c.copy()
    img_crop.save(path + 'img_crop.jpg')
    img_bw = img_c.copy()
    width, height = img_bw.size
    img_clean = img_c.copy()

#    img_clean remove the backgroud but have some noise
#    img_bw focus only on up triangle, used to define the work area
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

    img_clean.save(path + 'img_clean.jpg')
    img_bw.save(path + 'img_bw.jpg')

#    generation of edge image
    cv_img = numpy.array(img_clean)
    t_lower = 100  # Lower Threshold
    t_upper = 250  # Upper threshold
    cv_img = cv2.GaussianBlur(cv_img, (3,3), 0)
    edge = cv2.Canny(cv_img, t_lower, t_upper)
    img_final = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_final)
    im_edge_initial = im_pil.copy()
    im_edge_initial.save(path + 'im_edge_initial.jpg')

    width, height = img_bw.size
    finish = False
#    search of up triangle
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
    im_edge_crop = im_pil.copy()
    img_bw = img_bw.crop(area)
    img_bw_crop = img_bw.copy()
    img_bw_crop.save(path + 'img_bw_crop.jpg')
    im_edge_crop.save(path + 'im_edge_crop.jpg')

    width, height = im_pil.size
    finish = False

#    search of down triangle
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

#    search of triangle positions
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
            point_x.append(x1)
            point_y.append(y1)
            point_x.append(x2)
            point_y.append(y1)
        elif ( len(up_point) == 3 ):
            x1 = up_point[0] + 10
            x2 = up_point[2] - 10
            point_x.append(x1)
            point_y.append(y1)
            point_x.append(x2)
            point_y.append(y1)
        elif ( (len(up_point) == 4) ):
            x1 = round((up_point[1] + up_point[0]) / 2)
            x2 = round((up_point[3] + up_point[2]) / 2)
            point_x.append(x1)
            point_y.append(y1)
            point_x.append(x2)
            point_y.append(y1)
        else:
            print('Image   error with up triangles: ' + str(len(up_point)) )
            return False

        if ( len(down_point) == 1 ):
            if ( down_point[0] < 80 ):
                x = down_point[0] - 10
                if ( x < 0 ):
                    x = 0
            else:
                x = down_point[0] + 10
                if ( x >= width ):
                    x = width - 1
            point_x.append(x)
            point_y.append(y2)
        elif ( len(down_point) == 2 ):
            if ( (down_point[1] - down_point[0]) > 22 ):
                print('Image   error with down triangle: ' + str(len(down_point)) )
                return False
            x = round((down_point[1] + down_point[0]) / 2)
            point_x.append(x)
            point_y.append(y2)
        else:
            print('Image   error with down triangle: ' + str(len(down_point)) )
            return False
    else:
        print('Image   not found all triangles')
        return False
    finalx = []
    finaly = []

    print('Point_c: x: ' + str(point_x))
    print('         y: ' + str(point_y))

    for i in range(len(point_x)):
        finalx.append( point_x[i] + ref_min_x1 + ref_min_x2 + ref_min_x3 )
        finaly.append( point_y[i] + ref_min_y1 + ref_min_y2 + ref_min_y3 )
    if ( len(finalx) == 0):
        print('No triangle fuonds')
        return False
    print('finalx[0]: ' + str(finalx[0]))
    print('finalx[1]: ' + str(finalx[1]))
    print('width/2: ' + str(width/2))

    if (abs(point_x[0] - (width / 2)) <= abs(point_x[1] - (width / 2))):
        yellow.append(finalx[0])
        yellow.append(finaly[0])
        green.append(finalx[1])
        green.append(finaly[1])
        print('0 more near')
    else:
        yellow.append(finalx[1])
        yellow.append(finaly[1])
        green.append(finalx[0])
        green.append(finaly[0])
        print('1 more near')

    red.append(finalx[2])
    red.append(finaly[2])

    width, height = img.size
    img_final = img.copy()

    for i in range(height):
        img_final.putpixel((red[0],    i), (255,0,0))
        img_final.putpixel((yellow[0], i), (0, 0, 255))
        img_final.putpixel((green[0],  i), (0,255,0))

    img_final.save(path + 'img_final.jpg')

    return True


if __name__ == '__main__':

    path = '/home/gauss/projects/personal_ws/src/tests_package/foto_board'
    dir_list = os.listdir(path)

    for image_n in range(500):
        print(image_n)
        try:
            img  = Image.open('/home/gauss/projects/personal_ws/src/tests_package/foto_board/' + dir_list[image_n])
        except IOError:
            pass
        red = []
        yellow = []
        green = []

        if (not identification(img, red, yellow, green)):
            continue

        width, height = img.size

        for i in range(height):
            img.putpixel((red[0],    i), (255,0,0))
            img.putpixel((yellow[0], i), (0, 0, 255))
            img.putpixel((green[0],  i), (0,255,0))

        img.save(path + '__/Image' + str(image_n) + '.jpg')
    exit()
