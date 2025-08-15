# IndianArms-ObjDetection-Classification

Indian Arms span through a diversified set of groups that serve for the country. Every state has their own version of State Police that adds further types of Indian Civil service men, hence the need for varied uniforms. With varied uniforms there exists a need for proper knowledge on uniform identification. Automating this requires a two step analogy - Object Detection (Humans in Uniform) and Object Identification/Classification.

## Dataset-Description

Owing to restrictions on taking pictures of men on duty, there is a limited availability of all categories of personnel images inherently. Here we shall deal with 3 broad categories of service men - CRPF, BSF and JK (Jammu&Kashmir). Since we are working on a problem statement that requires simultaneous detection and classificaion, availability of proper annotated data further narrows down. Of all the publicly available ones, this repo utilises [data](https://universe.roboflow.com/dinesh-nariani-sakzd/unform-detection/dataset/13) because of it's integrity in terms of annotations and suitability to start model training. 

Problems encountered in this dataset:
  - **Lack of negative related samples (Type I)** 
  - **Lack of negative non-related samples (Type II)**

For Type I issue, the major problem that the model faced was ghost prediction (**False Positives**). Due to similar colour contrasts of other objects in the image background there was an increasing tendency to detect them. This energy was explainable since the model had to trade-off between **context**(shape) and **colour** information, situations when the colour information overshadowed contextual information the model steered towards this error. In order to solve this a new class was introduced as BG. BG stands for background (non-human) such that the default class of background coincides with it. Later the model produces a confusion matrix where a lot of these instances are predicted as background due to their inherent similarity. 35 new instances of 4th class **(BG)** were introduced in 35 different samples spread uniformly within 3 classes as a solution to Type I issue.

For Type II issue, challenges were totally different. Unrelated **tricky** data points that can challenge the softbed of training data were missing. Outsourcing these samples were important to make the model holistic in terms of real world deployment. Data points with description "military vehicle interior empty"/"camouflage pattern fabric". 60 samples were scraped off using a script with 15 different prompts and the whole of the image were annotated as **BG** class.

## Working Principle

Bounding boxes work on the core principle of maximising the probability for the object to lie inside the box. It becomes necessary for the detector to correctly indetify humans/partial human-like figure initially and map out a box around the region of interest (in this case a person wearing a uniform). Here we leverage YOLO frameworks with the specialisation of object detection and classification only. 

## Workflow 
Step by step solution/configs for data preparation and model training:

#### Train-Test split

--80:20 standard was followed to set aside 60 data points for final evaluation on unseen test data.

#### Data Augmentation

--Data availability as discussed is scarce. 302 samples across 3 classes is not enough for training the model from scratch hence **fine-tuning on a pre-trained model** remains the only viable option. Further elevation in data count is ensured by leveraging data augmentation techniques that add variance to original samples in the space of orientation/distribution/size. 

Following are the augmentation config:

    'imgsz': 640,       # INPUT SIZE  
    'hsv_h': 0.01,      # hue (HSV)
    'hsv_s': 0.4,       # saturation (HSV)
    'hsv_v': 0.3,       # brightness (HSV)
    'degrees': 10,      # max DEGREE OF ROTATION
    'translate': 0.1,   # TRANSLATION    
    'scale': 0.3,       # maximum SCALING FACTOR
    'flipud': 0.0,      # PROBABILITY OF UPSIDE DOWN FLIP
    'fliplr': 0.5,      # PROBABILITY OF LEFT RIGHT FLIP
    'mosaic': 0.5,      # mosaic TO COMBINE sub_regions of 4 diff IMAGES
    'mixup': 0.15       # mixup CRITERIA - PIXEL WISE COMBINATION

#### K-Fold Validation

--Instead of static train and valid splits, dynamic splitting becomes a non-negotiable criterion owing to small dataset. 5 fold cross validation exploits the possibility of best model churned out due to less variation captured in small sample size. Script for K-Fold splitting has been provided as well with custom yaml file generation. 

--Fold wise data distribution:

    | Metric                 | Fold 1     | Fold 2     | Fold 3     | Fold 4     | Fold 5     |
    | ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
    | **Training samples**   | 252        | 252        | 252        | 252        | 252        |
    | **Validation samples** | 28         | 28         | 28         | 28         | 28         |
    | **Training CRPF**      | 72 (28.6%) | 70 (27.7%) | 71 (28.1%) | 70 (27.8%) | 68 (26.9%) |
    | **Training BSF**       | 81 (32.1%) | 78 (30.8%) | 76 (30.1%) | 78 (31.1%) | 89 (35.3%) |
    | **Training JK**        | 67 (26.6%) | 70 (27.7%) | 73 (28.9%) | 72 (28.7%) | 65 (25.8%) |
    | **Training BG**        | 32 (12.7%) | 34 (13.7%) | 32 (12.9%) | 31 (12.4%) | 30 (12.1%) |
    | **Validation CRPF**    | 7 (24.4%)  | 8 (28.1%)  | 8 (26.7%)  | 8 (28.0%)  | 9 (31.9%)  |
    | **Validation BSF**     | 9 (31.1%)  | 10 (35.5%) | 11 (38.8%) | 10 (35.4%) | 5 (17.3%)  |
    | **Validation JK**      | 9 (31.6%)  | 8 (26.8%)  | 6 (22.3%)  | 6 (22.2%)  | 10 (35.1%) |
    | **Validation BG**      | 3 (13.0%)  | 2 (9.5%)   | 3 (12.1%)  | 4 (14.3%)  | 4 (15.7%)  |

### Model Development and Training

-- Models ranging from YoLoV8 to YoLo11 were tested on the 5 folds with varied data augmentation configs. After careful fold-wise inspection through the hyperparameter search spaces (warm-up epoch, close_mosaic, freeze_params, etc,.)
##### _YoLo11_ _fold4_ yield the best performing model 

Metrics
--


#### Fold-wise Metric table:
  
  
     Fold | mAP50 | mAP50-95 | Precision | Recall | AP50_CRPF  | AP50_BSF  | AP50_JK 
     ---- | ----- | -------- | --------- | ------ | ---------- | --------- | -------- 
     1    | 0.714 | 0.420    | 0.787     | 0.592  | 0.761      | 0.755     | 0.878    
     2    | 0.721 | 0.362    | 0.785     | 0.609  | 0.858      | 0.777     | 0.740    
     3    | 0.631 | 0.396    | 0.774     | 0.540  | 0.555      | 0.625     | 0.745    
     4    | 0.794 | 0.472    | 0.899     | 0.660  | 0.890      | 0.936     | 0.859    
     5    | 0.762 | 0.421    | 0.839     | 0.656  | 0.749      | 0.738     | 0.893   



  #### Metric Curves:

      
  <img width="1224" height="778" alt="Untitled design" src="https://github.com/user-attachments/assets/3a7c3ab6-e3c0-4ba5-a488-3ddb368b3645" />


  #### Confusion Matrix of Test Data:


  <img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/fe4f36fc-157c-439e-9393-a9435f5c19b3" />


  Model In Action
  --

  #### -> Unseen Sample_Same Dataset [Level - Individual]
  <img width="964" height="931" alt="Untitled design(1)" src="https://github.com/user-attachments/assets/f4ea5554-08e4-4c13-8c0e-f4f07008db87" />

  #### -> Unseen Sample_Same Dataset [Level - Group]
  <img width="853" height="905" alt="Untitled design(2)" src="https://github.com/user-attachments/assets/fac6d597-f5c1-4fec-82cf-8a3ed1480c40" />

  #### -> Different Dataset 
  Refer to **Lab's Result** dir for outputs on samples which are independently sourced from a different dataset (Heterogenous source)


  Usage
  --

  #### Dir Orgaization
    |
    |----Labs's
    |      |----BSF
    |      |----CRPF
    |      |----J&K
    |
    |----dataset_kfold_5
    |     |----Fold1
    |          |----train
    |              |----images
    |              |----labels
    |          |----valid
    |     |----Fold2
    |     |----Fold3
    |     |----Fold4
    |     |----Fold5
    |
    |----test
    |      |----images
    |      |----labels
    |
    |----train      
    |      |----images
    |      |----labels
    |
    |----weights
    |      |
    |      |----best.pt
    |      
    |
    |----get_bg_patches.py
    |----KFold_gen.py
    |----outsource_negatives.py
    |----train.py
    |----predict.py
  

  



  
        



    
    













































