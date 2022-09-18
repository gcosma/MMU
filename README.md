# Multi-Modal Unpairing Framework

### Introduction
This is the Python implementation for the Multi-Modal Unapiring Framework proposed in the paper ["A Multi-Modal Data Unpairing Framework for Cross-Modal Hashing Information Retrieval"](link). Within the aforementioned paper, the framework was utilsed alongside the methods DADH(https://github.com/Zjut-MultimediaPlus/DADH), AGAH(https://github.com/WendellGul/AGAH) and JDSH(https://github.com/KaiserLew/JDSH).
      
The flow of the MMU experiments is as follows:
* Prepare the datasets: (1) Download the datasets MIR-Flickr25K and NUS-WIDE. Information on where to download the datasets is provided in the "Download Data" section. (2) Place the datsets in the correct directories for each method. Information regarding the correct directory is provided within each method folder.
* Obtain the ranked loss for each training sample for the sample selection process of MMU, described in "Sample Selection" section. 
* Unpair the data through MMU, described in "Data Unpairing" section.
* Train and evaluate methods on newly generated unpaired data.

### Note
This is currently an informal version of the source code for MMU, subject to be formalised and made more accessible to readers pre-publishing.

### Requirements and installation
We recommended the following dependencies.
- Python 3.7.2
- Pytorch 1.6.0
- torchvision 0.7.0
- CUDA 10.1 and cuDNN 7.6.4

### Download Data

Data for the DADH method can be obtained from the [DADH repository](https://github.com/Zjut-MultimediaPlus/DADH).

Data for the AGAH and JDSH methods can be obtained from the [DCMH repository](https://github.com/jiangqy/DCMH-CVPR2017).

### Sample Selection

Once the dataset files have been downloaded and placed in the correct location for DADH, a standard training procedure is conducted on the altered main_samples.py script. This training procedure follows the parameters listed by [DADH](https://github.com/Zjut-MultimediaPlus/DADH)

### Data Unpairing
Once the datset files have been placed in their correct directory within MMU, run the visualize.py script. The default unpairing setting is set at 20% unpairing. To change this, follow the instructions within the source code of visualize.py.

### Training with unpaired data
Place the newly created unpaired dataset files into the correct datasets. Configure the method to be tested to point towards the newly created unpaired dataset file during data loading.
