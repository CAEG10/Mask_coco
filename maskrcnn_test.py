###MASKRCNN_PREDICT - ADAPTACION NP2020

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) #OMITIMOS WARNINGS

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

import numpy as np
import colorsys

import imutils
import random
import cv2
import os
#from matplotlib import pyplot as plt

weights='mask_rcnn_mosca_0010.h5'			#DATASET
labels='coco_labels.txt'				#ETIQUETAS QUE PUEDE DETECTAR
image='../imagenes/2.jpg'	#IMAGEN A PROBAR

CLASS_NAMES = open(labels).read().strip().split("\n")
#print(CLASS_NAMES) #objetos que se pueden detectar

cont=0  #contador de personas

####################################################################
class SimpleConfig(Config):
  NAME = "coco_inference"
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  NUM_CLASSES = len(CLASS_NAMES)

class_names = ['mosca']
###################MAIN
config = SimpleConfig()
print("CARGANDO Mask R-CNN ...")

model = modellib.MaskRCNN(mode="inference", config=config ,model_dir=os.getcwd())
#model.load_weights(weights, by_name=True)
#model.load_weights(weights_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
#model.load_weights(model_path, by_name=True)

image = cv2.imread(image)
cv2.imwrite("original.jpg", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=1024,height=780)

print("PREDICCIONES Mask R-CNN...")
r = model.detect([image], verbose=1)[0]

for i in range(0, r["rois"].shape[0]):
  classID = r["class_ids"][i]
  mask = r["masks"][:, :, i]
  color = (255,0,0) #RECTANGULOS VERDES  #COLORS[classID][::-1]

  #SE APLICA MASCARA A LA IMAGEN
  image = visualize.apply_mask(image, mask, color, alpha=0.5)

#CONVIERTE A BGR PARA USAR FUNCIONES DE DIBUJADO
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#####################################################
f=open('coord.txt','a')

image3 = cv2.imread('original.jpg',1)

for i in range(0, len(r["scores"])):

  (startY, startX, endY, endX) = r["rois"][i]
  classID = r["class_ids"][i]
  label = CLASS_NAMES[classID]
  score = r["scores"][i]
  color = (255,0,0) #[int(c) for c in np.array(COLORS[classID]) * 255]
  
  #############CONTADOR DE PERSONAS
  if label=='mosca':
    cont+=1
  
  #ENCIERRA CADA DETECCION DE OBJETOS EN RECTANGULOS
  cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

  ###################################
  #SE OBTIENEN LAS COORDENADAS DE LOS RECTANGULOS PARA 
  #EXPORTARLAS A ARCHIVO
  
  x=startX
  w=abs(endX-startX)
  y=startY
  h=abs(endY-startY)
  
  #crop_img = image3[y:y+h, x:x+w]
  #cv2.imshow("cropped"+str(i), crop_img)
  #cv2.imwrite("cropped"+str(i)+".jpg", crop_img)
  ###################################

  text = "{}: {:.3f}".format(label, score)
  #text = "{}: {:.3f}".format(startX,startY,endX,endY)

  txt = str(startX)+","+str(startY)+","+str(endX)+","+str(endY)+"\n"
  f.write(txt)

  y = startY - 10 if startY - 10 > 10 else startY + 10
  cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,0.6, color, 2)

f.close()

# MUESTRA RESULTADOS
cv2.imshow("Output", image)
cv2.imwrite('out.jpg', image)
cv2.waitKey()

if cont==0:
  print('NO MOSCAS DETECTADAS, HOjA SANA')
  
if cont==1:
  print('HAY '+str(cont)+' MOSCA')
  
if cont>1:
  print('HAY '+str(cont)+' MOSCAS')


