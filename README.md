**Warning:** This repository is outdated and it won't be maintained in future. Please refer to the repository https://github.com/boortel/AE-Reconstruction-And-Feature-Based-AD and to the ModelClassificationSIFT.py module for its direct Python reimplemantation. SURF based feature extractor will be added later.

# SIFT-and-SURF-based-AD
Implementation of the paper [SIFT and SURF based feature extraction for the anomaly detection](https://arxiv.org/abs/2203.13068)


Download the dataset at: https://www.kaggle.com/imonbilk/industry-biscuit-cookie-dataset

**EDIT:** New version of the dataset with the cropped images and simplified annotations is available as *Version 2*. Please use the script 
*DatasetFolder.py* attached to the dataset when using the updated version and the script in this repository for *Version 1*.

Download the Matlab SVDD code from: https://www.mathworks.com/matlabcentral/fileexchange/69296-support-vector-data-description-svdd and copy the Svdd folder to your working directory.


Please cite the following authors in your work:

```
@inproceedings{BUT177722,
  author="Šimon {Bilík} and Karel {Horák}",
  title="SIFT and SURF based feature extraction for the anomaly detection",
  address="Brno University of Technology, Faculty of Electrical Engineering",
  booktitle="Proceedings I of the 28 th Conference STUDENT EEICT 2022",
  chapter="177722",
  howpublished="online",
  institution="Brno University of Technology, Faculty of Electrical Engineering",
  year="2022",
  month="april",
  pages="459--464",
  publisher="Brno University of Technology, Faculty of Electrical Engineering"
}
```

```
@misc{Qiu2022,
  author = {Kepeng Qiu},
  journal = {GitHub},
  title = {Support Vector Data Description (SVDD)},
  subtitle = {MATLAB code for abnormal detection using Support Vector Data Description (SVDD)},
  year = {2022},
  medium = {online},
  accessed = {2022-10-11},
  URL = {https://github.com/iqiukp/SVDD-MATLAB/releases/tag/v2.1.5},
}
```
