# Multimodal-Information-Bottleneck (MIB)

News: We have uploaded a new version of the MIB paper, which addresses some mistakes and typos in the original pdf.

Here is the code for the paper "Multimodal Information Bottleneck: Learning Minimal Sufficient Unimodal and Multimodal Representations". The paper is accepted by IEEE Transactions on Multimedia, and the PDF is available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9767641.

Firstly, we would like to express our gratitude to the authors of MAG-BERT (https://github.com/WasifurRahman/BERT_multimodal_transformer). Their codes are of great help to our research.

To run the code, you firstly need to download the data using (see https://github.com/WasifurRahman/BERT_multimodal_transformer  for more details):

    sh datasets/download_datasets.sh

We have already provided the processed mosi data. For the larger mosei dataset, you should download it by the above command.

Then install the required packages using:

    pip install -r requirements.txt

Finally, we can run the codes using the following command:

To run C-MIB:

    python main_mib.py --mib cmib --dataset mosi --train_batch_size 32 --n_epochs 50

To run L-MIB:

    python main_mib.py --mib lmib --dataset mosi --train_batch_size 32 --n_epochs 50

To run E-MIB:

    python main_mib.py --mib emib --dataset mosi --train_batch_size 32 --n_epochs 50

To change the fusion methods, you can open the model files (cmib.py, emib.py, lmib.py), then go the fusion class, and then manually select the fusion methods. We provide six fusion methods for comparison: graph fusion (https://github.com/TmacMai/ARGF_multimodal_fusion), tensor fusion (https://github.com/Justin1904/TensorFusionNetworks), low-rank tensor fusion (https://github.com/Justin1904/Low-rank-Multimodal-Fusion), concat, addition, and multiplication. For tensor fusion, one should enlarge the d_l parameter in the MIB class for better performance. 

Currently two datasets are provided, i.e., mosi and mosei (please refer to https://github.com/A2Zadeh/CMU-MultimodalSDK for more details about the datasets). To run with mosei dataset, you should firstly open the global_configs.py, and then change the VISUAL_DIM to 47. We download the datasets by running (https://github.com/WasifurRahman/BERT_multimodal_transformer):

    pip install gdown

    gdown https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou

    gdown https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3. 


Finally, we can run the code using the following commands (change the mib parameter to your desired one):

    python main_mib.py --mib cmib --dataset mosei --train_batch_size 32 --n_epochs 50

If you find our codes useful, please cite our paper:

@ARTICLE{9767641,

  author={Mai, Sijie and Zeng, Ying and Hu, Haifeng},
  
  journal={IEEE Transactions on Multimedia}, 
  
  title={Multimodal Information Bottleneck: Learning Minimal Sufficient Unimodal and Multimodal Representations}, 
  
  year={2022},
  
  volume={},
  
  number={},
  
  pages={1-1},
  
  doi={10.1109/TMM.2022.3171679}}



