# Multimodal-Information-Bottleneck

Here is the code for the paper "Multimodal Information Bottleneck: Learning Minimal Sufficient Unimodal and Multimodal Representations" (MIB for multimodal sentiment analysis)

Firstly, we would like to express our gratitude to . Their codes are of great help to our research.

To run the code, you firstly need to download the data using (see  for more details):


Then install the required packages using:

Finally, we can run the codes using the following command:

To run C-MIB:

To run L-MIB:

To run E-MIB:

To change the fusion methods, you can open the model files (cmib.py, emib.py, lmib.py), then go the fusion class, and then manually select the fusion methods. We provide five fusion methods for comparison: tensor fusion, low-rank tensor fusion, graph fusion, concat, and multiplication.

Curreently two datasets are provided, i.e., mosi and mosei. To run with mosei dataset, you should firstly open the globalxx.py, and then change the TEXT_DIM parameter. Then, we can run the code using the following commands:

