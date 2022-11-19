## Using Multiple Instance Learning for Explainable Solar Flare Prediction

This github repository provides data, code and some results for the paper:

* Official version on Astronomy & Computing: https://authors.elsevier.com/sd/article/S2213-1337(22)00082-8
* Pre-print on arXiv: https://arxiv.org/abs/2203.13896

### Getting the data

The dataset is available via Zenodo: 

[IRIS Multiple Instance Learning Dataset](https://zenodo.org/record/6370336) (9.4 Gb, Numpy npz-Format)

### Running the code

The code was run on Python 3.6.9 and with the package versions listed in the requirements.txt file.
To run it, adhere to the following steps:

1. Create a virtual environment and install the required packages, e.g. with `virtualenv`:

```
virtualenv -p python3 irismil_env
source irismil_venv/bin/activate
pip install -r requirements.txt
```

2. Run the `model_runner.py` script with 

```
python model_runner.py <model_name> <parameter_value> <runs_per_fold>
```

e.g. to run an ibMIL model with r=3 and ten runs for each of the three CV-folds:

```
python model_runner.py ibMIL 3 10
```

### Videos (10 September 2014 Flare)

Pre-flare phase only:

https://user-images.githubusercontent.com/33490926/202858718-be83c44f-36c6-48a7-b599-c554ac9f552b.mp4


Whole observation:

https://user-images.githubusercontent.com/33490926/202858737-35c7cfc2-5f8d-4e80-a06e-4b68b0a07b12.mp4





