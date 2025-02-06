# CleaveNet

CleaveNet is an end-to-end AI pipeline for the design of protease substrates. Applied to matrix metalloproteinases, 
CleaveNet enhances the scale, tunability, and efficiency of substrate design. CleaveNet generates peptide substrates 
that exhibit sound biophysical properties and capture not only well-established but also novel cleavage motifs. To 
enable precise control over substrate design, CleaveNet incorporates a conditioning tag that enables generation of 
peptides guided by a target cleavage profile, enabling targeted design of efficient and selective substrates. 
CleaveNet-generated substrates were validated through a large-scale _in vitro_ screen, demonstrating success even 
in the challenging task of achieving high selectivity for MMP13. We envision that CleaveNet will accelerate our ability 
to study and capitalize on protease activity, paving the way for new _in silico_ design tools across enzyme
classes.

<p align="center">
<img src="img/fig1_predictor_only.png" width="290" />
<img src="img/fig1_generator_only.png" width="505" />
</p>

Below we provide documentation on training, evaluating, and generating from CleaveNet. 

### Citation
CleaveNet is described in this [preprint](https://www.biorxiv.org/content/); if you use the code from this repository or the results, please cite the preprint.

----
## Table of contents

- [CleaveNet](#cleavenet)
- [Table of contents](#table-of-contents)
- [Quick Setup](#quick-setup)
- [Getting Started](#getting-started-)
  - [Installation](#installation)
  - [Project structure](#project-structure)
- [Training models](#training-)
- [Predicting cleavage](#cleavenet-predictor)
- [Generating protease substrates](#cleavenet-generator-)
----

## Jupyter notebook
To get started quickly we have put together a [jupyter notebook](notebooks/CleaveNet_Example.ipynb) that simplifies using the code for substrate cleavage prediction, and unconditional substrate generation. We recommend starting here if you are less familiar with setting up and using a python package.


----
## Getting Started 
### Installation

To download our code, we recommend creating a clean environment with python ```v3.8```, and installing `tensorflow2`. 
```
conda env create --name cleavenet python=3.8
conda activate cleavenet 
```
Install CleaveNet using: 
```
pip install cleavenet
```
Or, for bleeding edge installation use: 
```
pip install git+https://github.com/microsoft/cleavenet.git # bleeding edge, current repo main branch
```

### Project structure
```
├── README.md
├── cleavenet/                          ~ package functions
├── data/                               
    ├── kukreja.csv                     ~ raw dataset as csv file           
    ├── cleavenet_design.txt            ~ sequences and controls tested; sequences only
    ├── cleavenet_design_summary.xlsx   ~ sequences and controls tested; MMP13 predicted Z-score, in vitro eff and selectivity
├── examples/                           ~ example data needed to run test code in README.md      
├── img/                                ~ images in README.md 
├── notebooks/                           
    ├── notebooks/                           
├── splits/                             ~ processed data splits for model training (output of DataLoader)
    ├── bhatia/                         ~ fluorescence test dataset  
    ├── kukreja/                        ~ mRNA display train/test splits 
    ├── kukreja_rounded/                ~ rounded splits for conditional generation training
├── src                                 ~ scripts for training and inference
    ├── generate.py                         
    ├── predict.py                          
    ├── train_generator.py                  
    ├── train_predictor.py              
├── weights                             ~ saved weights
    ├── {training_scheme}_{model_arc}   ~ generator models 
    ├── {model_arc}_{ensemble}          ~ predictor models 
```
----

## Training 
To train CleaveNet predictor: 

```
python src/train_predictor.py
```

To train CleaveNet generator: 
```
python src/train_generator.py
```

* The generator model is autoregressive trained over both conditional inputs (Z-scores included), and unconditional (sequence-only), next token prediction as described in the paper. 

### Datasets

By default, these scripts will use the training splits in `splits/` if present. If splits are not present it will create a new training split from `data/kukreja.csv`. This file was obtained from [Kukreja et. al., 2015](https://www.sciencedirect.com/science/article/pii/S1074552115002574?via%3Dihub) and converted to a csv file. If you want to use your own data, we suggest either converting your data into csv file formatted as `kukreja.csv` and passing it through the CleaveNet dataloader, or writing a compatible custom DataLoader. See `cleavenet/data.py` for an example.

----

## CleaveNet Predictor
### Substrate screeing with CleaveNet Predictor
You can use `src/predict.py` to predict cleavage of peptide sequences by our evaluated MMPs: MMP1, MMP2, MMP3, MMP7, MMP8, MMP9, MMP10, 
MMP11, MMP12, MMP13, MMP14, MMP15, MMP16, MMP17, MMP19, MMP20, and MMP25.

Run `src/predict.py` by pointing to a csv file that contains peptide sequences. Output plots and predictions will be saved to `--save-dir`

For example, the following line will evaluate the Kukreja test set seqeunces. 
```
python src/predict.py --path-to-sequence-csv splits/kukreja/X_test.csv --save-dir outputs/
```
The input csv should contain each sequence on a new line. Our model accepts sequences up to 10 residues in length. If sequences are shorter than 10 residues, pad them using a `-` token on both ends. 
For example shorter sequences could look like:
```
---AHA----
---PRVA---
--LRVFL---
--PRVVFLR-
PRVFQLRVFL
```

Corresponding true Z-scores for the input sequences can be passed to the script in a CSV file using
`--path-to-zscores scores.csv`
These scores are not be used for prediction, but can be used to run additional evaluations between the predicted and true cleavage behavior. Output files and plots will be saved to `--save-dir`. 

The following is an example of generating CleaveNet predictions, and comparing them to true Z-scores for the fluorescence test set. 
```
python src/predict.py --path-to-sequence-csv splits/bhatia/X_all.csv \
                      --path-to-zscores splits/bhatia/y_all_labeled.csv \
                      --save-dir outputs/bhatia_test
```

Where `y_all_labeled.csv` contains the corresponding Z-scores for `X_all.csv` and the first line is a header line corresponding to the MMPs of each column. It should look like; 
```
MMP1,MMP10,MMP12
-3.1,1.2,-2.1
4.5,2.6,-1.1
-1.5,2.1,-1.4
```

If you are interested in running predictions for the mRNA display test sequences evaluated in our paper you can use the following command 
```
python src/predict.py --path-to-sequence-csv splits/kukreja/X_test.csv \
                      --path-to-zscores splits/kukreja/y_test.csv \
                      --no-csv-header
                      --save-dir outputs/kukreja_test
```
The flag `--no-csv-header` assumes the csv file contains no header information, and will assign each column to the MMPs used to create the splits, and used during training. This flag is not necessary for most use cases, and should be used with caution.


**Notes:**  
* The model has not been robustly evaluated on sequences shorter or longer than 10 residues 
* The default CleaveNet predictor model uses a transformer. Using the Transformer default should be sufficient for most use cases. 
* In the paper we describe how in some cases an LSTM backbone can extrapolate better to unseen protease cleavage patterns. To instead use the baseline LSTM model for cleavage prediction add the flag `--model-architecture lstm` to `src/predict.py`. 

----

## CleaveNet Generator 
### Unconditional substrate generation
To generate protein sequences unconditionally using the CleaveNet generator run the following script: 
```
python src/generate.py --num-seqs 20000 --output-dir generations
```
The sampling temperature is controlled by the flag `--temperature`, smaller numbers will result in less diverse samples and higher numbers will increase sample diversity. 
A penalty can be applied to the frequency at which repeat amino acids occur by increasing the `--repeat-penalty` value. The default values for these are `1.0` and `1.2`, respectively, as used in evaluations for the CleaveNet paper. 
### Conditional substrate generation, given a cleavage profile
To conditionally generate sequences with a desired z-score profile, use the ``--z-scores`` flag. The following example will generate 400 sequences for each row provided in `examples/efficient_conditional_seeds.csv`, for a total of 20k samples. 
```
python src/generate.py --num-seqs 400 --output-dir conditional_generations --z-scores examples/efficient_conditional_seeds.csv
```
An example z-score file is given in `examples/efficient_conditional_seeds.csv`. This file should contain 18 columns with the MMPs labeled and a z-score corresponding to each. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
