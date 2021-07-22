#Put both labels and images in the same folder


#pip install clodsa
#The first step in the pipeline consists in loading the necessary libraries to apply the data augmentation techniques in CLODSA.
#Augmentation techniques
#we will use the following augmentation techniques:

#Vertical, horizontal, and vertical-horizontal flips.
#180º Rotation.
#Average blurring.
#Raise the hue value.

from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import xml.etree.ElementTree as ET
import cv2
#%matplotlib inline

#The kind of problem. In this case, we are working in a detection problem
PROBLEM = "detection"


#The annotation mode. We use the YOLO format.
ANNOTATION_MODE = "yolo"

#The input path. The input path containing the images.
INPUT_PATH = "DEEPSIGHT_YOLO"

#The generation mode. In this case, linear, that is, all the augmentation techniques are applied to all the images of the original dataset.
#Look into CLODSA documentation for different techniques
GENERATION_MODE = "linear"

#The output mode. The generated images will be stored in a new folder called augmented_images.
OUTPUT_MODE = "yolo"
OUTPUT_PATH= "augmented_images_yolo"

#Here we can create our augmentor object.
augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH})


#Now, we define the techniques that will be applied in our augmentation process and add them to our augmentor object.



#for test purposes we take a sample image
#img = cv2.imread("DEEPSIGHT_YOLO/train_00012650.jpg")
#plt.imshow(img[:,:,::-1])

#Just for showing the results of applying data augmentation in an object detection problem
#we define a function to read the annotations and another one to show them.
#This funcionality is not necessary when using CLODSA since it is already implemented in there.
#def boxesFromYOLO(imagePath,labelPath):
#    image = cv2.imread(imagePath)
#    (hI, wI) = image.shape[:2]
#    lines = [line.rstrip('\n') for line in open(labelPath)]
    #if(len(objects)<1):
    #    raise Exception("The xml should contain at least one object")
#    boxes = []
#    if lines != ['']:
#        for line in lines:
#            components = line.split(" ")
#            category = components[0]
#            x  = int(float(components[1])*wI - float(components[3])*wI/2)
#            y = int(float(components[2])*hI - float(components[4])*hI/2)
#            h = int(float(components[4])*hI)
#            w = int(float(components[3])*wI)
#            boxes.append((category, (x, y, w, h)))
#    return (image,boxes)
#categoriesColors = {11: (255,0,0),14:(0,0,255)}

#def showBoxes(image,boxes):
#    cloneImg = image.copy()
#    for box in boxes:
#        if(len(box)==2):
#            (category, (x, y, w, h))=box
#        else:
#            (category, (x, y, w, h),_)=box
#        if int(category) in categoriesColors.keys():
#            cv2.rectangle(cloneImg,(x,y),(x+w,y+h),categoriesColors[int(category)],5)
#        else:
#            cv2.rectangle(cloneImg,(x,y),(x+w,y+h),(0,255,0),5)
#    plt.imshow(cloneImg[:,:,::-1])

#img,boxes = boxesFromYOLO("DEEPSIGHT_YOLO/train_00012650.jpg","DEEPSIGHT_YOLO/train_00012650.txt")
#showBoxes(img,boxes)
# the above section was just for testing purposes (can comment this out while running the script)



#Define a transformer generator
transformer = transformerGenerator(PROBLEM)

#vertical flip
vFlip = createTechnique("flip",{"flip":0})
augmentor.addTransformer(transformer(vFlip))

#testing purpose ,, applying the transformation
#plt.figure()
#plt.title("Original")
#showBoxes(img,boxes)
#vFlipGenerator = transformer(vFlip)
#vFlipImg,vFlipBoxes = vFlipGenerator.transform(img,boxes)
#plt.figure()
#plt.title("Transformed")
#showBoxes(vFlipImg,vFlipBoxes)
# the above section was just for testing purposes (can comment this out while running the script)

#horizontal flip
hFlip = createTechnique("flip",{"flip":1})
augmentor.addTransformer(transformer(hFlip))

#testing purpose ,, applying the transformation
#plt.figure()
#plt.title("Original")
#showBoxes(img,boxes)
#hFlipGenerator = transformer(hFlip)
#hFlipImg,hFlipBoxes = hFlipGenerator.transform(img,boxes)
#plt.figure()
#plt.title("Transformed")
#showBoxes(hFlipImg,hFlipBoxes)
# the above section was just for testing purposes (can comment this out while running the script)

#Horizontal and vertical flip
hvFlip = createTechnique("flip",{"flip":-1})
augmentor.addTransformer(transformer(hvFlip))

#testing purpose ,, applying the transformation
#plt.figure()
#plt.title("Original")
#showBoxes(img,boxes)
#hvFlipGenerator = transformer(hvFlip)
#hvFlipImg,hvFlipBoxes = hvFlipGenerator.transform(img,boxes)
#plt.figure()
#plt.title("Transformed")
#showBoxes(hvFlipImg,hvFlipBoxes)
# the above section was just for testing purposes (can comment this out while running the script)

#rotation
rotate = createTechnique("rotate", {"angle" : 90})
augmentor.addTransformer(transformer(rotate))

#testing purpose ,, applying the transformation
#plt.figure()
#plt.title("Original")
#showBoxes(img,boxes)
#rotateGenerator = transformer(rotate)
#rotateImg,rotateBoxes = rotateGenerator.transform(img,boxes)
#plt.figure()
#plt.title("Transformed")
#showBoxes(rotateImg,rotateBoxes)
# the above section was just for testing purposes (can comment this out while running the script)

#Blurring
avgBlur =  createTechnique("average_blurring", {"kernel" : 5})
augmentor.addTransformer(transformer(avgBlur))

#testing purpose ,, applying the transformation
#plt.figure()
#plt.title("Original")
#showBoxes(img,boxes)
#avgBlurGenerator = transformer(avgBlur)
#avgBlurImg,avgBlurBoxes = avgBlurGenerator.transform(img,boxes)
#plt.figure()
#plt.title("Transformed")
#showBoxes(avgBlurImg,avgBlurBoxes)
# the above section was just for testing purposes (can comment this out while running the script)

#HUE Changer
hue = createTechnique("raise_hue", {"power" : 3.6})
augmentor.addTransformer(transformer(hue))

#testing purpose ,, applying the transformation
#plt.figure()
#plt.title("Original")
#showBoxes(img,boxes)
#hueGenerator = transformer(hue)
#hueImg,hueBoxes = hueGenerator.transform(img,boxes)
#plt.figure()
#plt.title("Transformed")
#showBoxes(hueImg,hueBoxes)
# the above section was just for testing purposes (can comment this out while running the script)

#Also keep the original picture
none = createTechnique("none",{})
augmentor.addTransformer(transformer(none))


#Applying the final augmentation process
augmentor.applyAugmentation()




