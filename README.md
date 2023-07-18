# PODOR
The goal of this project is to develop a robust and scalable solution that can automatically identify and classify potholes in real-time using image and video data from various sources, such as dashcams, traffic cameras, or drones.
![image](https://github.com/tschia94/PODOR/assets/139099352/07e5f517-ec30-4ec9-a41e-edd2d6d3a564)
# Pothole Severity determination
We determine the severity of a pothole based on its size
Area of bounding box for each pothole= ((ymax-ymin) *(xmax-xmin))
Take the area of the largest pothole in the image and compare it with a threshold. 
- Area > threshold -> Severity = High
- Area within threshold range -> Severity = Medium
- Area < threshold -> Severity = Low
- Threshold range was determined by running the tests on a batch of images.
- Number of potholes in an image with the score more than the threshold value (0.5)
# Proposed Architecture for downstream integration
![image](https://github.com/tschia94/PODOR/assets/139099352/9df73564-cebf-4a1a-8f4a-769bedd15757)
# Models (unable to upload models due to large file size)
- MaskRCNN
- DeeplabV3 ResNet101
# Credits
- The team would like to express gratitude to Datature for providing their platform for images annotation
