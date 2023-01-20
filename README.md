# Binary-Mask-to-COCO-Annotation-Format

## Introduction

There are many annotations format for different types of models. Usually, binary masks are used for U-NET and FPN model training. Suppose, if we want to train the MASK R-CNN model for instance segmentation using the same dataset, we need the COCO format annotations. This is the method we developed to convert binary mask to COCO annotation format.

## Installation process

Step 1: Clone this repository
```
!git clone https://github.com/Dilagshan/Binary-Mask-to-COCO-Annotation-Format.git
```
<br> Step 2: Change the path of the directory
```
cd /Binary-Mask-to-COCO-Annotation-Format
```
<br> Step 3: Install the requirement libraries 
```
!pip install -r requirement.txt
```
<br> Step 4: Import the file named " binary_to_coco.py"
```
from binary_to_coco import create_coco
```
<br> Step 5: Define the input for the function create_coco.
```
input_path = "list of binary images' path"
output_path = "path where you want to save the COCO annotations as json file"
create_coco(input_path, output_path)
```

## Reference
1. https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
