# radiation_dose_prediction

The Radiation Dose Prediction for Cancer Patients project, as part of the medical imaging MVA class, aims to predict the radiation dose cancer patients will receive during radiation therapy. The project focuses on developing a model that utilizes CT scans, dose masks, and segmentation masks of organs to accurately predict radiation doses. In the report, we present the models and techniques employed to achieve precise dose predictions. The most successful model achieved a MAE loss of 0.34 on the test set (0.03 behind the best model in the challenge), utilizing the **DCNN** model using dilated convolutions on the 2D CT-scans, masks of 10 organs, and dose masks of the region where radiation is possible. 

### Dataset 
* Comes from the Open-KBP challenge.
* Modified into a 2D dataset.
* Content of the folder of a sample:
  * Structural masks: binary masks of the 10 organs involved in the treatment
    * 10 x 128 x 128 volume
  * Possible dose mask: binary mask of where there irradiation is allowed
    * 128 x 128 image
  * Dose: ground-truth dose
    * 128 x 128 image
  * CT: scan of the patient
    * 128 x 128 image
    
 ![image](https://user-images.githubusercontent.com/79949319/235660260-e6eb8358-8479-43d8-a5d3-aceccb6830dd.png)

### Approach 
Among the presented methods and strategies, the best performant model used a U-NET like CNN with dilated convolution in order to widen the area of the input image covered without pooling. 






