## Using Multiple Instance Learning for Explainable Solar Flare Prediction

This github repository provides data, code and some results for the paper.

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

https://user-images.githubusercontent.com/33490926/150113373-f3e4b2a6-a0db-4fcf-8e3a-ba5aece3aca3.mp4

Whole observation:

https://user-images.githubusercontent.com/33490926/150113340-83140834-a1a0-4a4b-8920-2352de10455e.mp4

