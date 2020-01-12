
# HGFPN: 3D Facial Landmarks Detection for Intelligent Video Systems

## Prerequisite
- MXNet>=1.2.1
- tqdm==4.19.1
- Matplotlib
- NumPy
- scipy
- torchfile
- jupyter notebook

## Table of results
- LS3D-W: (Score format: ```AUC / Mean (StD) NME```)

    | Method | 300VW-3D-CatA | 300VW-3D-CatB | 300VW-3D-CatC | Menpo-3D | 300W-3D
    |:-:|:-:|:-:|:-:|:-:|:-:|
    | 3DDFA [1] | 56.51 / 3.29 \(2.83\) | 55.68 / 3.18 \(**2.01**\) | 39.24 / 4.88 \(3.99\) | 49.93 / 3.92 \(3.30\) | 53.15 / 3.62 \(3.13\) |
    | 3D-FAN [2] | 69.34 / 2.36 \(3.75\) | 70.54 / 2.31 \(3.93\) | 50.05 / 4.17 \(6.03\) | 65.54 / 2.38 \(**1.27**\) | **81.09** / **1.27** \(**0.37**\)
    **HGFPN** | **73.52** / **1.93** \(**2.12**\) | **74.63** / **1.94** \(**2.70**\) | **61.78** / **2.93** \(**3.48**) | **71.97** / **1.96** \(1.72\) | 78.89 / 1.43 \(0.42\)

- AFLW2000-3D

    | Method | AUC | [0, 30] | [30, 60] | [60, 90] | Mean (StD) |
    |:-:|:-:|:-:|:-:|:-:|:-:|
    | 3DDFA [1] | - | 4.11 | 4.38 | 5.16 | 4.55 \(**0.54**\)
    | 3DDFA + SDM [1] | - | 3.43 | 4.24 | 7.17 | 4.94 \(-\)
    | 3D-FAN [2] | - | 2.47 | 3.01 | 4.31 | 3.26 \(-\)
    | 3DDFA* [1] | 49.90 | 3.11 | 4.18 | 5.52 | 3.68 \(2.71\)
    | 3D-FAN* [2] | 57.66 | 2.51 | 3.27 | 4.46 | 2.95 \(1.47\)
    | **HGFPN** | **62.47** | **2.29** |  2.90 | 4.32 | **2.71** \(2.59\)
    
    <sup>* denotes results obtained by running public code.</sup>
    
- AFLW2000-3D-Reannotated

    | Method | AUC | [0, 30] | [30, 60] | [60, 90] | Mean (StD) |
    |:-:|:-:|:-:|:-:|:-:|:-:|
    | 3DDFA* [1] | 57.32 | 2.58 | 3.48 | 5.03 | 3.13 \(2.48\)
    | 3D-FAN* [2] | 72.69 | 1.85 | 1.84 | **2.24** | 1.91 \(**1.77**\)
    | **HGFPN** | **74.22** | **1.59** | **1.75** | 3.16 | **1.86** \(2.46\)

    <sup>* denotes results obtained by running public code.</sup>
    
## Run evaluation
### Prepare data
- You can download LS3D-W datasets from [homepage](https://www.adrianbulat.com/face-alignment) and ALFW-2000-3D from [hompage](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- Put LS3W-3D and AFLW2000-3D to ```data``` folder
- Modify path of datasets in ```get_preds.py``` file

### Run evaluation
- Run ```get_preds.py``` to get the predicted result of the proposed **HGFPN**.
- Run ```evaluation_all.ipynb``` to get the AUC, NME scores.
- Run ```eval_aflw2000.ipynb``` to get the detailed score of AFLW2000(-Reannotated).
- Run ```visualization.py``` to get the AUC curves.

### Output of methods
- Predicted results of 3DDFA (version 20190822) can be downloaded from [Google Drive](https://drive.google.com/open?id=1e_x_kbHcpmjBmAdSyclG-Gpen-A2LL_D) or can be obtained by run code from their [public code](https://github.com/cleardusk/3DDFA).
- Predicted results of 3D-FAN (version 20190822) can be downloaded from [Google Drive](https://drive.google.com/open?id=1wLzrrL1sad2jcCP8OyY7_tELid3VhAuo) or obtained by running code from their [public code](https://github.com/1adrianb/2D-and-3D-face-alignment).
    - You should modifiy the ```main.lua``` file. Line ```predictions[i] = preds_img:clone() + 1.75``` to ```predictions[i] = preds_img:clone()``` to obtain better results.
- Predicted results of HGFPN can be downloaded from [Google Drive](https://drive.google.com/open?id=1gNSyrGL5lkZGvELQrhRIjw16VzADytu0).

## Citation
    @article{thanh2020tii3dfacial,
        author={Van-Thanh Hoang, De-Shuang Huang, and Kang-Hyun Jo},
        journal={IEEE Transactions on Industrial Informatics},
        title={3D Facial Landmarks Detection for Intelligent Video Systems},
        year={2020},
        ISSN={1551-3203},
    }

## References
[1] X. Zhu, X. Liu, Z. Lei, and S. Z. Li, “Face alignment in full pose range: A 3d total solution,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 1, pp. 78–92, 2017.

[2] A. Bulat and G. Tzimiropoulos, “Hierarchical binary cnns for landmark localization with limited resources,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.
