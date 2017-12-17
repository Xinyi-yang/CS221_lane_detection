__author__ = 'taojia'
import json
from pprint import pprint



#
# eg_json = {"lanes": [[-2, -2, -2, -2, 632, 625, 617, 609, 601, 594, 586, 578, 570, 563, 555, 547, 539, 532, 524, 516, \
#             508, 501, 493, 485, 477, 469, 462, 454, 446, 438, 431, 423, 415, 407, 400, 392, 384, 376, 369, \
#             361, 353, 345, 338, 330, 322, 314, 307, 299], [-2, -2, -2, -2, 719, 734, 748, 762, 777, 791, \
#             805, 820, 834, 848, 863, 877, 891, 906, 920, 934, 949, 963, 978, 992, 1006, 1021, 1035, 1049, \
#             1064, 1078, 1092, 1107, 1121, 1135, 1150, 1164, 1178, 1193, 1207, 1221, 1236, 1250, 1265, -2, \
#             -2, -2, -2, -2], [-2, -2, -2, -2, -2, 532, 503, 474, 445, 416, 387, 358, 329, 300, 271, 241, 212, \
#             183, 154, 125, 96, 67, 38, 9, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, \
#             -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, 781, 822, 862, 903, 944, 984, 1025, 1066, 1107, \
#             1147, 1188, 1229, 1269, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, \
#             -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]],
# "h_samples": [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, \
#               430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, \
#               620, 630, 640, 650, 660, 670, 680, 690, 700, 710],
# "raw_file": "clips/0313-1/6040/20.jpg"}

#json_path = '~/Google Drive/Course/CS/CS221/Project/tuSimple_Label/'
def data2txt(data):
    print data
    #Goal: <object-class> <x> <y> <width> <height>
    output_folder = '/Users/taojia/Google Drive/Course/CS/CS221/Project/Code/darknet/VOCdevkit/VOC2012/labels/'
    #output_folder = 'VOCdevkit/LD/labels/'
    sec2last = data['raw_file'].split('/')[-2]
    last = data['raw_file'].split('/')[-1]
    output_name = sec2last + '_' + last[:-4] + '.txt'

    object_classes = ['1', '2', '3', '4', '5']
    print output_name
    #output_name = 'output.txt'
    image_width = 1280
    image_height = 720
    every_n_box = 5
    box_width = 10.0 * every_n_box
    box_height = 10.0 * every_n_box
    with open(output_folder + output_name, 'w') as f:
        width = box_width / image_width
        height = box_height / image_height
        lanes = data['lanes']
        for lane in range(len(lanes)):
            no_lane = lane
            for pixel in range(0, len(lanes[lane]), every_n_box):
                x = (lanes[lane][pixel] - box_width / 2) / image_width
                if x < 0:
                    continue
                y = (data['h_samples'][pixel] - box_height / 2) / image_height
                # If we want to separate lanes:
                # output_string = object_classes[no_lane] + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'

                # If we want to treat all the lanes as the same class:
                output_string = '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'
                f.write(output_string)



json_name = 'label_data_0601.json'
with open(json_name) as js:
    while True:
        line = js.readline()
        if line == '':
            break
        data = json.loads(line)
        data2txt(data)


pprint(data)