# some configuration
EPOCHS = 200 # upper limit of epochs
DATA_FILE = "IRISMIL_dataset_10000_bags.npz"
BASENAME = "full10" # basename used in stored results

# arguments
import os, sys
if len(sys.argv) != 4:
    print("""Usage: python abMIL_runner.py <model> <parameter> <runs_per_fold>
    model: ibMIL or abMIL
    parameter: hyperparameter r (ibMIL) or gamma (abMIL)
    runs_per_fold: number of trained models per CV fold (set to 10 in the paper)
    """)
    sys.exit(1)

# load all libraries (once we are sure to be in the correct environment)
import numpy as np
from tqdm.auto import tqdm
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from mil import ibMIL, abMIL

# set parameters
model = sys.argv[1]
parameter = float(sys.argv[2])
runs_per_fold = int(sys.argv[3])

# check parameters
if model not in ['ibMIL', 'abMIL']:
    print("Please use one of the models 'ibMIL' or 'abMIL'")
    sys.exit(1)

if model == 'ibMIL':
   parameter_name = 'r'
else:
   parameter_name = 'gamma'

# print info
print("Training with {} runs per fold for {} model with {} = {}".format(runs_per_fold, model, parameter_name, parameter))
print("-" * 80)

# prepare data
print("\n\n\n==> Preparing data..")
f = np.load( DATA_FILE, allow_pickle=True )
data_scaled = f['data_scaled']
masks = f['masks']
groups = f['groups']
obs_ids = f['obs_ids']
obs_classes = f['obs_classes']
folds = f['folds']

X = data_scaled
classmap = {'AR': 0, 'PF': 1}
y = np.array( [classmap[t] for t in obs_classes] )
X = np.expand_dims( X, axis=-1 )

# create folds
print("==> Creating folds..")
folds += 1
np.unique( folds, return_counts=True )
fold1 = (folds == 1)
fold2 = (folds == 2)
fold3 = (folds == 3)
test1 = fold1
train1 = fold2 | fold3
test2 = fold2
train2 = fold3 | fold1
test3 = fold3
train3 = fold1 | fold2

# create directories for models and training curves if they do not exist yet
if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("curves"):
    os.mkdir("curves")

# ibmil runner
def ibmil_runner( basename, rs, train_folds, test_folds, epochs=200, lr=0.00001, n=3, loss="mean_absolute_error" ):
    
    for r in tqdm(rs):
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||| r = {} ||||||||||||||||||||||||||||||||||||||||||||||||||\n".format(r))
        
        for f in tqdm(range(len(train_folds))):
            print("=================================================== fold {} =========================================================".format(f))
            
            for i in range(n):
                print("----------------------------------[ RUN {} ]-----------------------------------------------".format(i))
    
                instance_MIL = ibMIL( 
                    X.shape[1:], pooling="logmeanexp", r=r,
                    conv_units=[],
                    dense_units=[(50,0.2,0)],
                    flat_dropout=0.0,
                    final_kernel_regularizer=l2(0),
                    model_name="{}_r{}_run{}_fold{}".format( basename, r, i, f ), 
                    summary=False
                )

                log = instance_MIL.train( 
                    (X[train_folds[f]], masks[train_folds[f]]), y[train_folds[f]], (X[test_folds[f]], masks[test_folds[f]]), y[test_folds[f]], lr=lr,
                    epochs=epochs, batch_size=32, shuffle=True, verbose=2, optimizer=Adam, early_stopping=True, loss=loss,
                    metrics=['accuracy', 'Precision', 'Recall'], plot_metrics=True,
                    plot_metrics_savepath="curves/ibmil_" + instance_MIL.model_name.replace('.', '')
                )


                # save model
                print("==> Saving model")
                instance_MIL.save( savedir="models" )
                
                # predict on data and save predictions
                print("==> Predicting")
                y_train_prob = instance_MIL.predict_bags( (X[train_folds[f]], masks[train_folds[f]]) )[:,0]
                y_test_prob = instance_MIL.predict_bags( (X[test_folds[f]], masks[test_folds[f]]) )[:,0]
                np.savez("models/ibmil_{}_pred.npz".format(instance_MIL.model_name), y_train_prob=y_train_prob, y_test_prob=y_test_prob, y_train=y[train_folds[f]], y_test=y[test_folds[f]])
                print("\n")
                
            print("\n")
                
        print("\n\n")


# abmil runner
def abmil_runner( basename, gammas, train_folds, test_folds, epochs=200, lr=0.00001, n=3, loss="mean_absolute_error", multi_gpu=False ):
    
    for gamma in tqdm(gammas):
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||| gamma = {} ||||||||||||||||||||||||||||||||||||||||||||||||||\n".format(gamma))
        
        for f in tqdm(range(len(train_folds))):
            print("=================================================== fold {} =========================================================".format(f))
            
            for i in range(n):
                print("----------------------------------[ RUN {} ]-----------------------------------------------".format(i))
    
                attention_MIL = abMIL( 
                    X.shape[1:], M=10, L=10, heads=1, gamma=gamma,
                    emb_conv_units=[],
                    emb_dense_units=[(50,0.2,0)],
                    clf_dense_units=[],
                    flat_dropout=0.0,
                    final_kernel_regularizer=l2(0),
                    model_name="{}_gamma{}_run{}_fold{}".format( basename, gamma, i, f ), 
                    summary=False
                )

                log = attention_MIL.train( 
                    (X[train_folds[f]], masks[train_folds[f]]), y[train_folds[f]], (X[test_folds[f]], masks[test_folds[f]]), y[test_folds[f]], lr=lr,
                    epochs=epochs, batch_size=32, shuffle=True, verbose=2, optimizer=Adam, early_stopping=True, loss=loss,
                    early_stopping_args={'monitor': 'val_loss', 'patience': 20, 'min_delta': 0.001, 'restore_best_weights': True},
                    metrics=['accuracy', 'Precision', 'Recall'], plot_metrics=True,
                    plot_metrics_savepath="curves/abmil_" + attention_MIL.model_name
                )

                # save model
                print("==> Saving model")
                attention_MIL.save( savedir="models" )
                
                # predict on data and save predictions
                print("==> Predicting")
                y_train_prob = attention_MIL.predict_bags( (X[train_folds[f]], masks[train_folds[f]]) )[:,0]
                y_test_prob = attention_MIL.predict_bags( (X[test_folds[f]], masks[test_folds[f]]) )[:,0]
                np.savez("models/abmil_{}_pred.npz".format(attention_MIL.model_name), y_train_prob=y_train_prob, y_test_prob=y_test_prob, y_train=y[train_folds[f]], y_test=y[test_folds[f]])
                print("\n")
                
            print("\n")
                
        print("\n\n")
        
        
        
# actual run
print("==> Starting runner..")

if model == 'ibMIL':
    ibmil_runner( 
        basename=BASENAME, rs=[parameter], train_folds=[train1,train2,train3], test_folds=[test1,test2,test3], epochs=EPOCHS,
        lr=0.00001, n=runs_per_fold, loss="mean_absolute_error"
    )

else:
    abmil_runner( 
        basename=BASENAME, gammas=[parameter], train_folds=[train1,train2,train3], test_folds=[test1,test2,test3], epochs=EPOCHS,
        lr=0.00001, n=runs_per_fold, loss="mean_absolute_error"
    )