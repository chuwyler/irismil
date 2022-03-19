import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, TimeDistributed, Input, Lambda, Softmax, LeakyReLU, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler


# MIL Pooling Layer
class MILPoolingLayer( Layer ):

    def __init__( self, pooling="logmeanexp", r=None, L=None, M=None, heads=1, gamma=1, **kwargs ):
        self.pooling = pooling
        self.r = r
        self.L = L
        self.M = M
        self.heads = heads
        self.gamma = gamma
        
        # general variables needed for attention mechanism
        if "attention" in pooling:
            self.V = []
            self.w = []
            
            for i in range(heads):
                w_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform') 
                V_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform') 
                self.w.append( tf.Variable( initial_value=w_init(shape=(L,1)), dtype="float32", trainable=True ) )
                self.V.append( tf.Variable( initial_value=V_init(shape=(M,L)), dtype="float32", trainable=True ) )

        # additional variables needed for gating mechanism
        if pooling == "gated_attention":
            self.U = []
            
            for i in range(heads):
                U_init = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform') 
                self.U.append( tf.Variable( initial_value=U_init(shape=(M,L)), dtype="float32", trainable=True ) )

                
        super(MILPoolingLayer, self).__init__( **kwargs )
    
    def call( self, x ):
        x, masks = x
    
        if self.pooling == "logmeanexp":
            n_unmasked = K.expand_dims( K.sum(masks, axis=1), axis=-1 )
            exp_sum = K.sum(K.expand_dims(masks, axis=-1) * K.exp(self.r*x), axis=1)
            return 1/self.r * K.log( 1/n_unmasked * exp_sum )
        
        elif self.pooling == "mean":
            return 1/K.expand_dims( K.sum(masks, axis=1), axis=-1) * K.sum(K.expand_dims(masks, axis=-1) * x, axis=1)
            
        elif self.pooling == "max":
            return K.max(K.expand_dims(masks, axis=-1) * x, axis=1)

        elif self.pooling == "attention":
            emb = []
            att = []
            for i in range(self.heads):
                a = K.expand_dims(masks, axis=-1) * K.dot( K.tanh( K.dot(x, self.V[i]) ), self.w[i] )
                a = K.expand_dims(masks, axis=-1) * K.exp(a)
                a = a / K.expand_dims( K.sum(a, axis=1), axis=1 )
                a = a**self.gamma / K.expand_dims( K.sum(a**self.gamma, axis=1), axis=1 )
                att.append(a)
                emb.append(K.sum( a * x, axis=1 ))
            
            if len(emb) > 1:
                emb = Concatenate(axis=1)(emb)
            else:
                emb = emb[0]
                
            return att, emb
            
        elif self.pooling == "gated_attention":
            emb = []
            att = []
            for i in range(self.heads):
                a = K.expand_dims(masks, axis=-1) * K.dot( K.tanh( K.dot(x, self.V[i]) ) * K.sigmoid(K.dot(x, self.U[i])), self.w[i] )
                a = K.expand_dims(masks, axis=-1) * K.exp(a)
                a = a / K.expand_dims( K.sum(a, axis=1), axis=1 )
                a = a**self.gamma / K.expand_dims( K.sum(a**self.gamma, axis=1), axis=1 )
                att.append(a)
                emb.append(K.sum( a * x, axis=1 ))
            
            if len(emb) > 1:
                emb = Concatenate(axis=1)(emb)
            else:
                emb = emb[0]
                
            return att, emb
        
        else:
            raise Exception("This pooling layer is not available")
            
    # override get_config such that models with this layer can be stored
    def get_config( self ):
        config = super().get_config().copy()
        config.update({
            'pooling': self.pooling,
            'L': self.L,
            'M': self.M,
            'heads': self.heads,
            'gamma': self.gamma
        })
        if type(self.r) is int:
            config.update({'r': self.r})
        else:
            config.update({'r': None})
        
        return config


# convolutional unit shortcut
def conv_unit(x, filters, width, strides, l2_reg, **kwargs):
    x = Conv1D( filters, width, strides, padding="same", kernel_regularizer=l2(l2_reg), **kwargs )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.9)(x)
    return x

# dense unit shortcut
def dense_unit(x, nodes, dropout=0.5, l2_reg=0, **kwargs):
    x = Dense(nodes, kernel_regularizer=l2(l2_reg), **kwargs)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(dropout)(x)
    return x

# increase train epochs callback (for counting)
class TrainEpochIncrement(Callback):
    def on_train_begin(self, logs=None):
        self.train_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        self.train_epochs += 1
        
# MIL baseclass
class MIL:
    def __init__( self, model_type, model_name, summary=True ):
        self.model_type = model_type
        self.model_name = model_name
        self.summary = summary
        self.train_epochs = None
        
        self._create_graph()
            
        if self.summary: print( self.bag_classifier.summary() )
   
    # this is particular to abMIL or ibMIL 
    def _create_graph( self ):
        raise NotImplementedError
        
    def train( 
        self, X_train, y_train, X_test, y_test, lr=0.001, loss="binary_crossentropy", metrics=['accuracy'], 
        optimizer=Adam, optimizer_args={}, early_stopping=True, early_stopping_args={'monitor': 'val_loss', 'patience': 5, 'min_delta': 0.005, 'restore_best_weights': False}, 
        plot_metrics=False, lr_schedule=None, save_checkpoints=False, plot_metrics_savepath=None, **kwargs 
    ):
        
        # delete all saved epochs
        if save_checkpoints:
            try:
                os.system("rm {}-{}-*".format(self.model_type, self.model_name))
            except:
                pass
        
        # compile model
        compile_kwargs = {"loss": loss, "optimizer": optimizer(learning_rate=lr, **optimizer_args), "metrics": metrics}
        self.bag_classifier.compile( **compile_kwargs )
        
        # add desired callbacks
        train_epoch_increment = TrainEpochIncrement()
        callbacks = [train_epoch_increment]
        if save_checkpoints:
            callbacks.append( ModelCheckpoint('models/'+self.model_type+'-'+self.model_name+'-{epoch:02d}.h5') )
        if early_stopping:
            callbacks.append(EarlyStopping(**early_stopping_args))
        if lr_schedule is not None:
            callbacks.append(LearningRateScheduler(lr_schedule, verbose=1))
            
        # run training
        log = self.bag_classifier.fit( X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks, **kwargs )
        if plot_metrics: plot_training(log, savepath=plot_metrics_savepath)
        
        # set training epochs
        self.train_epochs = train_epoch_increment.train_epochs

        return log

    # load one of the stored epochs again (if checkpoints are saved)
    def load_epoch( self, epoch ):
        saved_models = pd.Series(os.listdir("models"))
        available_epochs = sorted( saved_models[saved_models.str.contains("{}-{}-".format(self.model_type, self.model_name))].str[-5:-3].astype(int).values )
        if len(available_epochs) == 0:
            raise Exception("No checkpoints saved for this model!")
        if epoch < 0: # allow cycling
            epoch = np.max(available_epochs) + epoch + 1
        self.bag_classifier.load_weights('models/{}-{}-{:02d}.h5'.format(self.model_type, self.model_name, epoch))

    # predict bag labels on input bags
    def predict_bags( self, *args, **kwargs ):
        return self.bag_classifier.predict( *args, **kwargs )
    
    # save model in combination with its parameters (not trivial - this could be improved)
    def save( self, savedir="." ):
        # save model class
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        members = {member: getattr(self, member) for member in members}
        del members['model_type']
        params_savepath = "{}/{}_{}_params.json".format( savedir, self.model_type, self.model_name )
        with open(params_savepath, 'w') as json_file:
            json.dump(members, json_file)
        print("Saved class parameters in {}".format(params_savepath))
       
        # save weights
        model_savepath = "{}/{}_{}_model".format( savedir, self.model_type, self.model_name )
        self.bag_classifier.save( model_savepath )
        print("Saved model in {}".format(model_savepath))
    
    # load a saved model graph and the model parameters    
    @classmethod
    def load( cls, model_name, savedir="." ):
        
        # determine model type
        model_type = cls.__name__.lower()
        #print("model type: {}".format(model_type))
        
        # load model
        param_savepath = "{}/{}_{}_params.json".format( savedir, model_type, model_name )
        with open(param_savepath, 'r') as json_file:
            init_parameters = json.load(json_file)
            train_epochs = init_parameters['train_epochs']
            del init_parameters['train_epochs']
            ret = cls( **init_parameters )
            ret.train_epochs = train_epochs
       
        model_savepath = "{}/{}_{}_model".format( savedir, model_type, model_name ) 
        ret.bag_classifier = load_model( model_savepath )
            
        return ret

# instance-based MIL class
class ibMIL(MIL):
    def __init__( 
        self, data_shape, pooling="logmeanexp", r=1, conv_units=[], dense_units=[], flat_dropout=0.3, final_kernel_regularizer=l2(0), 
        training_noise=0, summary=False, model_name="0" 
    ):
        self.data_shape = data_shape
        self.pooling = pooling
        self.r = r
        self.conv_units = conv_units
        self.dense_units = dense_units
        self.flat_dropout = flat_dropout
        self.final_kernel_regularizer = final_kernel_regularizer
        self.training_noise = training_noise
        super().__init__( model_type="ibmil", model_name=model_name, summary=summary  )
        
    def _create_graph( self ):
        instance_input = Input( shape=self.data_shape[1:] )
        x = instance_input
        
        # convolutional units
        for cu in self.conv_units:
            x = conv_unit( x, cu[0], cu[1], cu[2], cu[3] )
        
        if self.summary: print("Shape before flattening:", x.shape)
        x = Flatten()(x)
        if self.summary: print("Shape after flattening:", x.shape)
        x = Dropout(self.flat_dropout)(x)
        
        # dense units
        for du in self.dense_units:
            x = dense_unit( x, du[0], du[1], du[2] )
        
        # predict single instance
        x = Dense(1, activation="sigmoid",kernel_regularizer=self.final_kernel_regularizer)(x)
        instance_prediction = Model( instance_input, x )
        
        # masks
        masks = Input( shape=(self.data_shape[0],) )
        
        # apply simple classifier to whole input and pool
        bag_input = Input( shape=self.data_shape )
        instance_probs = TimeDistributed( instance_prediction )( bag_input )
        self.instance_classifier = Model( bag_input, instance_probs )
        self.pooling_layer = MILPoolingLayer( pooling=self.pooling, r=self.r )
        bag_aggregate = self.pooling_layer( [instance_probs, masks] )
        self.bag_classifier = Model( [bag_input, masks], bag_aggregate )
        
    def predict_instances( self, X, mask, verbose=True, **kwargs ):
        pred = self.instance_classifier.predict( X, **kwargs, batch_size=32, verbose=verbose )
        return pred * np.expand_dims(mask, axis=-1)
    
# attention-based MIL class
class abMIL(MIL):
    def __init__( 
        self, data_shape, L, M, heads=1, gamma=1, emb_conv_units=[], emb_dense_units=[], clf_dense_units=[], flat_dropout=0.3, final_kernel_regularizer=l2(0),
        summary=False, model_name="0" 
    ):
        self.data_shape = data_shape
        self.L = L
        self.M = M
        self.heads = heads
        self.gamma = gamma
        self.emb_conv_units = emb_conv_units
        self.emb_dense_units = emb_dense_units
        self.clf_dense_units = clf_dense_units
        self.flat_dropout = flat_dropout
        self.final_kernel_regularizer = final_kernel_regularizer
        super().__init__( model_type="abmil", model_name=model_name, summary=summary  )
        
    def _create_graph( self ):
        instance_input = Input( shape=self.data_shape[1:] )
        x = instance_input
        
        # convolutional units for embedding
        for cu in self.emb_conv_units:
            x = conv_unit( x, cu[0], cu[1], cu[2], cu[3] )
        if self.summary: print("Shape before flattening:", x.shape)
        x = Flatten()(x)
        if self.summary: print("Shape after flattening:", x.shape)
        x = Dropout(self.flat_dropout)(x)
        
        # dense units
        for du in self.emb_dense_units:
            x = dense_unit( x, du[0], du[1], du[2] )
        
        # predict embedding
        x = Dense(self.M)(x)
        embedding = Model(instance_input, x)
        
        # masks
        masks = Input( shape=(self.data_shape[0],) )
        
        # pooling with attention
        bag_input = Input( shape=self.data_shape )
        embeddings = TimeDistributed( embedding )( bag_input )
        self.instance_embedding = Model( bag_input, embeddings )
        self.pooling_layer = MILPoolingLayer( pooling="gated_attention", L=self.L, M=self.M, heads=self.heads, gamma=self.gamma )
        a, bag_aggregate = self.pooling_layer( [embeddings, masks] )
        self.bag_embedding = Model( [bag_input, masks], bag_aggregate )
        self.attention = Model( [bag_input, masks], a )
        
        # predict on aggregated embedding
        bag_prediction = bag_aggregate
        for du in self.clf_dense_units:
            bag_prediction = dense_unit( bag_prediction, du[0], du[1], du[2] )
        bag_prediction = Dense(1, activation="sigmoid")( bag_prediction )
        self.bag_classifier = Model( [bag_input, masks], bag_prediction )

    def predict_embedding( self, *args, verbose=True, **kwargs ):
        return self.instance_embedding.predict( *args, **kwargs, batch_size=32, verbose=verbose )
    
    def predict_bag_embedding( self, *args, verbose=True, **kwargs ):
        return self.bag_embedding.predict( *args, **kwargs, batch_size=32, verbose=verbose )
    
    def predict_attention( self, X, masks, verbose=True, normalize=True, **kwargs ):
        att = self.attention.predict( (X, masks), **kwargs, batch_size=32, verbose=verbose )
        if self.heads == 1:
            att = [att]
        
        if normalize:
            for head in range(len(att)):
                att[head] *= np.expand_dims( masks, axis=-1 ) # set masked values to zero
                N = np.sum(masks, axis=1) # rescale such that mean is 1
                att[head] *= N.reshape(-1,1,1)
        
        return att
    

# function to plot training evolution
def plot_training( log, show=True, add_log_scale_loss=True, figsize=(25,3), savepath=None ):
    # extract metrics
    metrics = [k for k in log.history.keys() if not 'val' in k]
    n = len(metrics)
    plt.figure(figsize=figsize)
    if add_log_scale_loss: n+=1
    for i, m in enumerate(metrics):
        plt.subplot(1,n,i+1)
        plt.plot( log.history[m], label="training" )
        if 'val_'+m in log.history.keys():
            plt.plot( log.history['val_'+m], label="test" )
        plt.title( m )
        plt.legend()
    
    if add_log_scale_loss:
        plt.subplot(1,n,n)
        m = "loss"
        plt.plot( log.history[m], label="training" )
        if 'val_'+m in log.history.keys():
            plt.plot( log.history['val_'+m], label="test" )
        plt.title( m + " (log scale)" )
        plt.legend()
        plt.yscale("log")

    plt.tight_layout()

    if savepath is not None:
        plt.savefig( savepath + ".jpg" )

    if show:
        plt.show()
