
# HGFPN: 3D Facial Landmarks Detection for Intelligent Video Systems

## Prerequisite
- MXNet>=1.2.1
- tqdm==4.19.1
- Matplotlib
- NumPy
- scipy
- torchfile
- jupyter notebook

## Evaluation
### Preparation
- You can download LS3D-W datasets from ```https://www.adrianbulat.com/face-alignment``` and ALFW-2000-3D from ```http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm```
- Put LS3W-3D and AFLW2000-3D to ```data``` folder
- Modify path of datasets in ```get_preds.py``` file

### Run evaluation
- Run ```get_preds.py``` to get the predicted result of the proposed HGFPN
- Run ```evaluation_all.ipynb``` to get the AUC, NME scores.
- Run ```eval_aflw2000.ipynb``` to get the detailed score of AFLW2000(-Reannotated)
- Run ```visualization.py``` to get the AUC curves

### Output of methods
- Predicted results of 3DDFA (version 20190822) can be downloaded from ```https://drive.google.com/open?id=1e_x_kbHcpmjBmAdSyclG-Gpen-A2LL_D``` or can be obtained by run code from ```https://github.com/cleardusk/3DDFA```
- Predicted results of 3D-FAN (version 20190822) can be downloaded from ```https://drive.google.com/open?id=1wLzrrL1sad2jcCP8OyY7_tELid3VhAuo``` or obtained by running code from ```https://github.com/1adrianb/2D-and-3D-face-alignment```
    - You should modifiy the ```main.lua``` file. Line ```predictions[i] = preds_img:clone() + 1.75``` to ```predictions[i] = preds_img:clone()``` to obtain better results.
- Predicted results of HGFPN can be downloaded from