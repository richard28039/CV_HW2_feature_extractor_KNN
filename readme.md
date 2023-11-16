# Method description 

## Gabor Filter
- python main.py --train ./plant-seedings-classification/train/ --feature_extractor Gabor_Filters

- python main.py --test ./plant-seedings-classification/test/ --feature_extractor Gabor_Filters  

-  python main.py --knn_predict ./Gabor_Filters/ --feature_extractor Gabor_Filters

## Local Binary Pattern
- python main.py --train ./plant-seedings-classification/train/ --feature_extractor Local_Binary_Pattern

- python main.py --test ./plant-seedings-classification/test/ --feature_extractor Local_Binary_Pattern  

- python main.py --knn_predict ./Local_Binary_Pattern/ --feature_extractor Local_Binary_Pattern

## Color histogram
- python main.py --train ./plant-seedings-classification/train/ --feature_extractor Color_histogram  

- python main.py --train ./plant-seedings-classification/test/ --feature_extractor Color_histogram  

- python main.py --knn_predict ./Color_histogram/ --feature_extractor Color_histogram
- 
## raw image
- python main.py --knn_predict ./plant-seedings-classification/ --feature_extractor Raw_image

# Experimental results

![Alt text](image.png)

![Alt text](image-1.png)

![Alt text](image-2.png)

![Alt text](image-4.png)

resnet50
![Alt text](image-3.png) 

# Discussion

Color histogram may be the best feature extractor in this case beacause of the leaf size.

# Problem and difficulties
Currently not

