import cv2
import json
import numpy as np

def find_annotations(path,id=0):
  annotations = []
  binary_mask = cv2.imread(path,0)
  name = path.split('/')[-1]
  image_id = name.split('_')[-1].split('.')[0]
  _, thresh_binary_mask = cv2.threshold(binary_mask,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  #contours = measure.find_contours(thresh_binary_mask, 0.5)
  contours, _= cv2.findContours(thresh_binary_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  id = id
  for contour in contours:
    anno =  { 'id': id,
              'iscrowd': 0,
              'image_id': int(image_id),
              'category_id': 0 ,
              'segmentation': [], 
              'bbox' : [],
              'area' : cv2.contourArea(contour)
              }
    id = id + 1

    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
    n = approx.ravel().tolist()

    contour = np.flip(contour, axis=1)
    anno["segmentation"].append(n)
    x,y,w,h = cv2.boundingRect(contour)
    anno["bbox"] = [x,y,w,h]
    annotations.append(anno)
  return id, annotations

def create_images(path):
  dic = {}
  name = path.split('/')[-1]
  image_id_ = name.split('_')[-1].split('.')[0]
  binary_mask = cv2.imread(path,0)
  dic['id'] = int(image_id_)
  dic['width'] = binary_mask.shape[1]
  dic['height'] = binary_mask.shape[0]
  dic['file_name'] = name
  return dic

def create_coco(path_list,output_path):
  data = {'info':'', 
          'images':[], 
          'annotations':[], 
          'categories':[{"id": 0, "name": "my class" }]}
  data['info'] = {"description": "Binary to COCO"}
  for i in range(len(path_list)):
    data['images'].append(create_images(path_list[i]))
    if i == 0:
      id, annotations = find_annotations(path_list[i])
      for i in annotations:
        data['annotations'].append(i)
    else:
      id, annotations = find_annotations(path_list[i], id = id)
      for i in annotations:
        data['annotations'].append(i)

  output_json = json.dumps(data,indent = 4) # save the dictionary in readble json format

  # write our output in a json file
  with open (output_path,'w') as f:
    f.write(output_json)
    f.close()

