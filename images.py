import os
import cv2
dir = './mnist_test/test'
images = os.listdir(dir)
images.sort()
test_dataset = []
labels = []
for i in range(len(images)):
    imgpath = os.path.join(dir, images[i])
    images[i] = images[i][:-4]
    imgarray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    imgarray = cv2.resize(imgarray, (28, 28))
    imgarray = imgarray
    test_dataset.append(imgarray)
    imglabel = int(images[i].split('-')[1][-1])
    labels.append(imglabel)

# write to file
with open('test_dataset.txt', 'w') as f:
    for i in range(len(test_dataset)):
        for j in range(28):
            for k in range(28):
                f.write(str(test_dataset[i][j][k]) + '\n')
f.close()
# write labels
with open('test_labels.txt', 'w') as f:
    for i in range(len(labels)):
        f.write(str(labels[i]) + '\n')
f.close()
    
    
    
    

