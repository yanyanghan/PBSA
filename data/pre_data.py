import json
import os
import shutil
data = './images'
new = './train'
images = os.listdir(data)
jsontext = {'images':[]}
for image in images:
    for line in open('labels.txt'):
        a = line.split()
        b = a[0]
        if b.split('.')[0] == image.split('.')[0]:
            c = a[1]
            with open('./synsets.txt', 'r') as ss:
                for num, lines in enumerate(ss):
                    if num == int(c):
                        lines = lines.strip()
                        print(lines)
                        if os.path.exists(new + '/' + lines) is False:
                            os.mkdir(new + '/' + lines)
                        source = data + '/' + str(image)
                        deter = new + '/' + lines + '/' + image.split('.')[0] + '.JPEG'
                        shutil.copyfile(source, deter)
                        image_new = lines + '/' + image.split('.')[0] + '.JPEG'
images_1 = os.listdir(new)
for image_1 in images_1:
    name = os.listdir(new+'/'+image_1)
    name = name[0]
    jsontext['images'].append(image_1+'/'+name)
jsondata = json.dumps(jsontext, indent=4)
f = open('image_list.json', 'w')
f.write(jsondata)
f.close()
