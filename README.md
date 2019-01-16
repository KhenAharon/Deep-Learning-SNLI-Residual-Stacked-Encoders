This is a reimplemention code for residual stacked encoders based on the original code by the authors (https://arxiv.org/pdf/1708.02312.pdf).<br>
The original code by the authors cannot currently run due to missing files and runtime errors.

## Important notes to be aware of:<br>
A. You need an NVIDIA gpu with more than 2GB of ram and cuda available to run the original code by the authors after handling missing files and runtime errors.<br>
B. You can use my code to run on the CPU, but you still need a super hardware computer with at least 32GB of general RAM.<br>
C. Using residual connection instead of shorcut connection may reduce the hardware requirment, but I didn't try it.

## Create the "data" directory. 
create "data" directory in the root directory.
Add to the data directory the following files:

1. "saved_embd.pt" from my drive:
https://drive.google.com/open?id=1vDJfjEUGnK-S3gZ5sv6Q_PDeJ0P3tR6q

or create it by following my comment on the issue here:
https://github.com/easonnie/multiNLI_encoder/issues/5

2. extract the text file "glove.840B.300d.txt" to the directory-
http://nlp.stanford.edu/data/glove.840B.300d.zip

3. copy snli_1.0 directory from this link :
https://nlp.stanford.edu/projects/snli/snli_1.0.zip
(extract and copy the directory)

4. do the same for multinli_0.9 from here:
*add link from google.

## make up your environment:
```
sudo apt-get update
sudo apt-get install python-dev
sudo apt-get install python3-dev
```

## install the dependencies:
```
torch
fire
tqdm
numpy
torchtext (version 0.1.1)
```

## install spacy by pip:
```
pip install -U spacy
python -m spacy download en
```

## The files hierarchy should look like:
```
.
├── config.py
├── data
│   ├── multinli_0.9
│   │   ├── multinli_0.9_dev_matched.jsonl
│   │   ├── multinli_0.9_dev_mismatched.jsonl
│   │   ├── multinli_0.9_test_matched_unlabeled.jsonl
│   │   ├── multinli_0.9_test_mismatched_unlabeled.jsonl
│   │   └── multinli_0.9_train.jsonl
│   ├── saved_embd.pt
│   └── snli_1.0
│       ├── README.txt
│       ├── snli_1.0_dev.jsonl
│       ├── snli_1.0_dev.txt
│       ├── snli_1.0_test.jsonl
│       ├── snli_1.0_test.txt
│       ├── snli_1.0_train.jsonl
│       └── snli_1.0_train.txt
├── model
│   └── res_encoder.py
├── saved_model
│   └── trained_model_will_be_saved_in_here.txt
├── setup.sh
├── torch_util.py
└── util
    ├── data_loader.py
    ├── dataset_util.py
    ├── mnli.py
    └── save_tool.py
```

## Start training by run the script in the root directory.
```
source setup.sh
python model/res_encoder.py train_snli
```
then the model will be saved in the saved_model directory.

## Now, you can evaluate the model on dev set again by running the script below.
```
python model/res_encoder.py eval (PATH_OF_YOUR_MODEL) dev # for evaluation on dev set
python model/res_encoder.py eval (PATH_OF_YOUR_MODEL) test # for evaluation on test set
```

