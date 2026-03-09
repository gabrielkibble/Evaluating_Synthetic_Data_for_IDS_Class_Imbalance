""" IDS contains the functions necessary to create, train, and test an autoencoder
    based intrusion detection system. 
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

from numpy.random import seed

#--------------------------------------------------------------------------------
def train(data_set, data_encoding, hyper_param):
    """Creates and trains an autoencoder.  

       Arguments:
        data_set -> Dataframe of data to be used for testing.

        data_encoding -> instance of cDataEncoding class defining how data_set is encoded

        hyper_param -> instance of the cHyperParam class defining hyperparmaters to be used
    """
    seed(hyper_param.random_seed)
    tf.random.set_seed(hyper_param.random_seed)

    print("\nTraining()")
    print("\nLoading data...")

# --- DIAGNOSTIC START ---
    print(f"DEBUG: Drop List is: {data_encoding.COL_FOR_SCORING}")
    
    # 1. Attempt the drop
    # We use errors='ignore' so it doesn't crash if 'Timestamp' is missing
    X_train = data_set.drop(data_encoding.COL_FOR_SCORING, axis=1, errors='ignore')

    # 2. Verify exactly what remains
    print(f"\nDEBUG: Columns remaining AFTER drop ({len(X_train.columns)} total):")
    print(X_train.columns.tolist())
    
    # 3. Hunt for the "Object" (Text) columns
    print("\nDEBUG: Checking data types for every column...")
    problem_found = False
    for col in X_train.columns:
        # Check the type
        dtype = X_train[col].dtype
        
        # If it is 'object' (string/text), this is what is crashing Keras
        if dtype == 'object':
            print(f"  !!! PROBLEM FOUND !!! Column '{col}' is type '{dtype}'")
            print(f"      Sample value: '{X_train[col].iloc[0]}'")
            problem_found = True
            
    if problem_found:
        print("\n!!! CRITICAL ERROR: Text columns still exist. Training will fail.")
        # We raise an error here to stop execution so you can read the logs above
        raise ValueError("Text data found in training set (see logs above).")
    else:
        print("\nDEBUG: All columns are numeric. Proceeding...")

    # 4. Final conversion to ensure float32
    X_train = X_train.astype('float32')
    # --- DIAGNOSTIC END ---

    print(X_train.describe(include='all'))
    print("\nCreating model...")

    model = create_model(X_train.shape[1], hyper_param) 

    print("\nFitting model...")
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=hyper_param.early_term_period)

    history = model.fit(np.array(X_train),np.array(X_train),
                       batch_size=hyper_param.batch_size, 
                       epochs=hyper_param.num_epochs,
                       validation_split=0.05,
                       verbose = 1,
                       callbacks=[es])

    return model

#--------------------------------------------------------------------------------
def create_model(input_size, hyper_param):
    """ Creates a model using the provided hyperparamaters

        Arguments:
            input_size - number of inputs provided to network
            hyper_param - instance of cHyperParameters defining hyperparamters to be used
    """

    #TODO: add paramater for kernel and activity regularized
    #TODO: add paramater for activation function
    #TODO: Try KL-Divergence Sparse Autoencoder
    act_func = 'elu'

    # Input layer:
    model=Sequential()

    #ENCODER
    # First hidden layer, connected to input vector X
    model.add(Dropout(hyper_param.drop_out[0], input_shape=(input_size,)))

    model.add(Dense(hyper_param.layers[0],
                    activation=act_func,
                    kernel_initializer=initializers.glorot_uniform(seed=hyper_param.random_seed),
                    activity_regularizer=regularizers.l1(hyper_param.kernel_reg[0]),
                    #kernel_regularizer=regularizers.l2(kernel_reg[i]), 
                    ))

    model.add(Dropout(hyper_param.drop_out[1]))
    #print("Encoder: Added layer -> " + str(input_size) + ":" + str(hyper_param.layers[0]) + " k_reg: " + str(hyper_param.kernel_reg[0]) + " drop_out: " + str(hyper_param.drop_out[0]))

    for i in range(1,len(hyper_param.layers)-1):
        model.add(Dense(hyper_param.layers[i],
                        activation=act_func,
                        #kernel_regularizer=regularizers.l2(kernel_reg[i]),
                        activity_regularizer=regularizers.l1(hyper_param.kernel_reg[i]),
                        kernel_initializer=initializers.glorot_uniform(seed=hyper_param.random_seed)))
        model.add(Dropout(hyper_param.drop_out[i+1]))
        #print("Encoder: Added layer -> " + str(hyper_param.layers[i]) + " k_reg: " + str(hyper_param.kernel_reg[i]) + " i is " + str(i))


    #BOTTLENECK
    model.add(Dense(hyper_param.layers[-1],
                    activation=act_func,
                    #kernel_regularizer=regularizers.l2(kernel_reg[-1]),
                    kernel_initializer=initializers.glorot_uniform(seed=hyper_param.random_seed)))
    #print("Bottleneck: Added layer - nodes: " + str(hyper_param.layers[-1]) + " k_reg: " + str(hyper_param.kernel_reg[-1]))

    #DECODER
    for i in range(len(hyper_param.layers)-2,-1,-1):
        model.add(Dense(hyper_param.layers[i],
                        activation=act_func,
                        #kernel_regularizer=regularizers.l2(kernel_reg[i]),
                        activity_regularizer=regularizers.l1(hyper_param.kernel_reg[i]),
                        kernel_initializer=initializers.glorot_uniform(seed=hyper_param.random_seed)))
        #print("Decoder: Added layer -> " + str(hyper_param.layers[i]) + " k_reg: " + str(hyper_param.kernel_reg[i]) + " i is " + str(i))

    model.add(Dense(input_size,
                    #kernel_regularizer=regularizers.l2(0.001),
                    kernel_initializer=initializers.glorot_uniform(seed=hyper_param.random_seed)))

    #print("Decoder: Added layer -> " + str(input_size) + " k_reg: " + str(hyper_param.kernel_reg[-1]))
    
    #TODO: Check for other loss functions
    #https://medium.com/@syoya/what-happens-in-sparse-autencoder-b9a5a69da5c6
    model.compile(loss='mse',optimizer='adam')
    
    print("\nModel Summary...")
    print(model.summary())

    return model

#--------------------------------------------------------------------------------
def test(model, test_data, data_encoding):  
    print("\nLoading data...")
    test_data.reset_index(drop=True, inplace=True)

    # 1. Clean column names (remove hidden spaces)
    test_data.columns = test_data.columns.str.strip()

    # 2. Drop the label columns
    drop_cols = ['Label', 'Full Label', 'Timestamp']
    test_data = test_data.drop(drop_cols, axis=1, errors='ignore')

    # 3. Force numeric conversion
    for col in test_data.columns:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    test_data = test_data.fillna(0.0)
    test_data = test_data.astype('float32')

    # --- THE FIX: Truncate to match Training Shape ---
    # The model expects 43 columns. If we have more, cut the end off.
    expected_cols = model.input_shape[1] # Should be 43
    if test_data.shape[1] > expected_cols:
        print(f"\nFIX: Truncating extra columns. Keeping first {expected_cols}...")
        test_data = test_data.iloc[:, :expected_cols]
    # -------------------------------------------------

    print("\nPredicting...")
    pred_data = model.predict(test_data)
    
    pred_data = pd.DataFrame(pred_data, columns=test_data.columns)
    pred_data.index = test_data.index
    pred_data['Loss'] = np.mean(np.abs(pred_data - test_data), axis=1)

    return pred_data

#--------------------------------------------------------------------------------
def apply_thresh(test_data, pred_data, threshold, ):
    """ Apply a threshold and calculate how well it identifies anomolies
    """
    print("\nScoring results...")

    scored = pd.DataFrame(index=test_data.index)
    scored['Threshold'] = threshold
    scored['Anomaly'] = pred_data['Loss'] > scored['Threshold']
    scored['Label'] = test_data['Label']
    scored['Full Label'] = test_data['Full Label']

    scored['True Negative'] = np.where((scored['Anomaly'] == False) & (scored['Label'] == False), True, False)
    scored['True Positive'] = np.where((scored['Anomaly'] == True) & (scored['Label'] == True), True, False)
    scored['False Positive'] = np.where((scored['Anomaly'] == True) & (scored['Label'] == False), True, False)
    scored['False Negative'] = np.where((scored['Anomaly'] == False) & (scored['Label'] == True), True, False)

    return scored

#--------------------------------------------------------------------------------
def calc_stats(scored, pred_data, verbose=True):
    """ Calculate a variety of statistical results from result
    """
    stats = {}
    stats['Total Anom'] = scored[scored['Label'] == True].count()['Label']
    stats['Total Norm'] = scored[scored['Label'] == False].count()['Label']

    stats['Avg Norm Score'] = scored[scored['Label'] == False].mean()['Label']
    stats['STD Norm Score'] = scored[scored['Label'] == False].std()['Label']
    stats['Avg Anom Score'] = scored[scored['Label'] == True].mean()['Label']
    stats['STD Anom Score'] = scored[scored['Label'] == True].std()['Label']

    #TODO: Only need to compare with a single value instead of two extremly long columns
    scored['Anom Above Avg Norm'] = np.where((pred_data['Loss'] >= stats['Avg Norm Score']) & (scored['Label'] == True), True, False)
    scored['Anom Above 1STD Norm'] = np.where((pred_data['Loss'] >= (stats['Avg Norm Score'] + 1 * stats['STD Norm Score'])) & (scored['Label'] == True), True, False)
    scored['Anom Above 2STD Norm'] = np.where((pred_data['Loss'] >= (stats['Avg Norm Score'] + 2 * stats['STD Norm Score'])) & (scored['Label'] == True), True, False)
    scored['Anom Above 3STD Norm'] = np.where((pred_data['Loss'] >= (stats['Avg Norm Score'] + 3 * stats['STD Norm Score'])) & (scored['Label'] == True), True, False)
    scored['Norm Less Avg Anom'] = np.where((pred_data['Loss'] <= stats['Avg Anom Score']) & (scored['Label'] == False), True, False)
    scored['Norm Less 1STD Anom'] = np.where((pred_data['Loss'] <= (stats['Avg Anom Score'] - 1 * stats['STD Anom Score'])) & (scored['Label'] == False), True, False)
    scored['Norm Less 2STD Anom'] = np.where((pred_data['Loss'] <= (stats['Avg Anom Score'] - 2 * stats['STD Anom Score'])) & (scored['Label'] == False), True, False)
    scored['Norm Less 3STD Anom'] = np.where((pred_data['Loss'] <= (stats['Avg Anom Score'] - 3 * stats['STD Anom Score'])) & (scored['Label'] == False), True, False)

    if(stats['Total Anom'] > 0):
        stats['Pcnt False Negative'] = 100*scored['False Negative'].sum()/stats['Total Anom']
        stats['Pcnt True Positive'] = 100*scored['True Positive'].sum()/stats['Total Anom']

        stats['Pcnt Anom Above Avg Norm']  = 100*scored['Anom Above Avg Norm'].sum()/stats['Total Anom']
        stats['Pcnt Anom Above 1STD Norm'] = 100*scored['Anom Above 1STD Norm'].sum()/stats['Total Anom']
        stats['Pcnt Anom Above 2STD Norm'] = 100*scored['Anom Above 2STD Norm'].sum()/stats['Total Anom']
        stats['Pcnt Anom Above 3STD Norm'] = 100*scored['Anom Above 3STD Norm'].sum()/stats['Total Anom']
    else:
        stats['Pcnt False Negative'] = 0
        stats['Pcnt True Positive'] = 0

        stats['Pcnt Anom Above Avg Norm']  = 0
        stats['Pcnt Anom Above 1STD Norm'] = 0
        stats['Pcnt Anom Above 2STD Norm'] = 0
        stats['Pcnt Anom Above 3STD Norm'] = 0

    if(stats['Total Norm'] > 0):
        stats['Pcnt True Negative'] = 100*scored['True Negative'].sum()/stats['Total Norm']
        stats['Pcnt False Positive'] = 100*scored['False Positive'].sum()/stats['Total Norm']

        stats['Pcnt Norm Below Avg Anom']  = 100*scored['Norm Less Avg Anom'].sum()/stats['Total Norm']
        stats['Pcnt Norm Below 1STD Anom'] = 100*scored['Norm Less 1STD Anom'].sum()/stats['Total Norm']
        stats['Pcnt Norm Below 2STD Anom'] = 100*scored['Norm Less 2STD Anom'].sum()/stats['Total Norm']
        stats['Pcnt Norm Below 3STD Anom'] = 100*scored['Norm Less 3STD Anom'].sum()/stats['Total Norm']
    else:
        stats['Pcnt True Negative'] = 0
        stats['Pcnt False Positive'] = 0

        stats['Pcnt Norm Below Avg Anom']  = 0
        stats['Pcnt Norm Below 1STD Anom'] = 0
        stats['Pcnt Norm Below 2STD Anom'] = 0
        stats['Pcnt Norm Below 3STD Anom'] = 0

    if(verbose):
        print_score(scored, stats)

    return scored, stats

#--------------------------------------------------------------------------------
def print_score(scored, stats, write_file=False, filename="test_output.txt"):
    """Prints stastical data from the scored dataframe output of 'test'.  .  

       Used for helping identify proper network design/parameters.
    
       Arguments:
        scored -> Dataframe with line-by-line scoring

        statistics -> Dataframe with summary statistics of the scoring

        write_file -> Boolean if want to write these results to a file (useful for
            looping runs trying wide variety of parameters).

        filename -> filename of text file where data is stored 
    """
    print("\n\nAnalyzing test data...")
    #print(scored.nlargest(20, ['Loss_mae']))

    #Scoring statistics
    print("\n\nTest set had:" + str(scored[scored['Label'] == False].count()['Label']) + " benign and " + str(scored[scored['Label'] == True].count()['Label']) + " attacks.")
    print("Result set has:" + str(scored[scored['Anomaly'] == False].count()['Anomaly']) + " benign and " + str(scored[scored['Anomaly'] == True].count()['Anomaly']) + " attacks.")

    print('\nAnomolies above average normal score: ' + str(scored['Anom Above Avg Norm'].sum()) + " "
           + str(stats['Pcnt Anom Above Avg Norm']))
    print('Anomolies above 1STD normal score: ' + str(scored['Anom Above 1STD Norm'].sum()) + " "
           + str(stats['Pcnt Anom Above 1STD Norm']))
    print('Anomolies above 2STD normal score: ' + str(scored['Anom Above 2STD Norm'].sum()) + " "
           + str(stats['Pcnt Anom Above 2STD Norm']))
    print('Anomolies above 3STD normal score: ' + str(scored['Anom Above 3STD Norm'].sum()) + " "
           + str(stats['Pcnt Anom Above 3STD Norm']))

    print('Normals less average anomaly score: ' + str(scored['Norm Less Avg Anom'].sum()) + " "
           + str(stats['Pcnt Norm Below Avg Anom']))
    print('Normals less 1STD anomaly score: ' + str(scored['Norm Less 1STD Anom'].sum()) + " "
           + str(stats['Pcnt Norm Below 1STD Anom']))
    print('Normals less 2STD anomaly score: ' + str(scored['Norm Less 2STD Anom'].sum()) + " "
           + str(stats['Pcnt Norm Below 2STD Anom']))
    print('Normals less 3STD anomaly score: ' + str(scored['Norm Less 3STD Anom'].sum()) + " "
           + str(stats['Pcnt Norm Below 3STD Anom']))
    print("\nAvg/std: anamoly = " + str(stats['Avg Norm Score']) + "/" + str(stats['STD Norm Score']) + 
              " normal = " + str(stats['Avg Anom Score']) + "/" + str(stats['STD Anom Score']) + "\n")

    print('True Negatives: ' + str(scored['True Negative'].sum()) + " " + str(stats['Pcnt True Negative']))
    print('True Positive: ' + str(scored['True Positive'].sum())  + " " + str(stats['Pcnt True Positive']))
    print('False Positive: ' + str(scored['False Positive'].sum()) + " " + str(stats['Pcnt False Positive']))
    print('False Negative: ' + str(scored['False Negative'].sum()) + " " + str(stats['Pcnt False Negative']))

    if(write_file):
        output_file = open(filename, "a") 

        output_file.write('\nAnomolies above average normal score: ' + str(scored['Anom Above Avg Norm'].sum()) + " "
               + str(stats['Pcnt Anom Above Avg Norm']))
        output_file.write('Anomolies above 1STD normal score: ' + str(scored['Anom Above 1STD Norm'].sum()) + " "
               + str(stats['Pcnt Anom Above 1STD Norm']))
        output_file.write('Anomolies above 2STD normal score: ' + str(scored['Anom Above 2STD Norm'].sum()) + " "
               + str(stats['Pcnt Anom Above 2STD Norm']))
        output_file.write('Anomolies above 3STD normal score: ' + str(scored['Anom Above 3STD Norm'].sum()) + " "
               + str(stats['Pcnt Anom Above 3STD Norm']))

        output_file.write('Normals less average anomaly score: ' + str(scored['Norm Less Avg Anom'].sum()) + " "
               + str(stats['Pcnt Norm Below Avg Anom']))
        output_file.write('Normals less 1STD anomaly score: ' + str(scored['Norm Less 1STD Anom'].sum()) + " "
               + str(stats['Pcnt Norm Below 1STD Anom']))
        output_file.write('Normals less 2STD anomaly score: ' + str(scored['Norm Less 2STD Anom'].sum()) + " "
               + str(stats['Pcnt Norm Below 2STD Anom']))
        output_file.write('Normals less 3STD anomaly score: ' + str(scored['Norm Less 3STD Anom'].sum()) + " "
               + str(stats['Pcnt Norm Below 3STD Anom']))
        output_file.write("\nAvg/std: anamoly = " + str(stats['Avg Norm Score']) + "/" + str(stats['STD Norm Score']) + 
                          " normal = " + str(stats['Avg Anom Score']) + "/" + str(stats['STD Anom Score']) + "\n")

        output_file.write('True Negatives: ' + str(scored['True Negative'].sum()) + " " + str(stats['Pcnt True Negative']))
        output_file.write('True Positive: ' + str(scored['True Positive'].sum())  + " " + str(stats['Pcnt True Positive']))
        output_file.write('False Positive: ' + str(scored['False Positive'].sum()) + " " + str(stats['Pcnt False Positive']))
        output_file.write('False Negative: ' + str(scored['False Negative'].sum()) + " " + str(stats['Pcnt False Negative']))

        output_file.write("\n\n")
        output_file.write(str(scored.nlargest(50, ['Loss_mae'])))
        output_file.write("\n")
        output_file.write(str(scored.nsmallest(20, ['Loss_mae'])))
        output_file.write("\n")
        output_file.close()

    print("Analysis complete.")

#--------------------------------------------------------------------------------
def save_anom(scored, raw_pickle, save_pickle):
    """Uses anomolies found during a test run to createa a pickle of only anomoly raw data.  

       Intention is to use this to potentially train a second layer of anomaly
       detection.  Creates a pickle of the raw parametric data for all rows which were
       flagged as an anomaly by the first layer.
    
       Arguments:
        results -> Dataframe output of a test run.  Should include a boolean column 
        'Anomaly' that is true if an error was detected.

        raw_pickle -> filename of pickle of raw data

        save_pickle -> filename of the new pickle to create (subset of the raw data)
    """
    print("\n\nSaving anomalies...")
    raw_data = pd.read_pickle(raw_pickle)
    raw_data.reset_index(drop=True, inplace=True)
    scored.reset_index(drop=True, inplace=True)
    raw_data['Anomaly'] = scored['Anomaly']
    anom = raw_data[raw_data['Anomaly'] == True]  
    anom.drop(columns=['Anomaly'], axis=1, inplace=True)
    
    print(anom.describe(include='all'))
    
    anom.to_pickle(save_pickle)

#--------------------------------------------------------------------------------
def flag_by_type(scored, write_file=False, filename="test_flag_by_type.txt"):
    """Group results by each individual anomoly.  Used to examine which anomoly types
       aren't being found by the network
    """
    full_labels = scored.groupby(['Full Label'])
    print(full_labels.Anomaly.value_counts())

    #TODO: Do a way to save it to a file or in a more organized table

#--------------------------------------------------------------------------------
def find_best_thresh(model, data_encoding, test_data, increments=50):
    """Increment through a range of potential threshold values to find best performing. 

       For now these results are just being printed for manual inspection.

       Arguments:
        model - training model to analyze
        X_test - test data to be used

    """
    #TODO: 
    # 1. Does the best true score also always have the best false score? If so can shrink
    #    this function down.
    # 2. Could save all results to a csv to see the overall solution space

    print("\n*** Finding best threshold ***")
    pred_data = test(model, test_data, data_encoding)

    losses = pd.DataFrame(index=test_data.index)
    
    best_score = 0;
    best_thresh = 0;
    best_results = []

    #Calculate incremental value by using difference of max and min MAE
    #min = pred_data['Loss'].min()
    #max = pred_data['Loss'].max()
    mean = pred_data['Loss'].mean()
    std = pred_data['Loss'].std()
    minimum = max(0, mean - 3*std)
    maximum = min(1, mean + 3*std)
    increment = (maximum-minimum)/increments
    print("Threshold range to be tested:: " + str(minimum) + ":" + str(increment) + ":" + str(maximum))

    for i in np.arange(minimum,maximum,increment):
        print("\n\n * Testing threshold: " + str(i) + " *")
        scored = apply_thresh(test_data, pred_data, i)
    
        final_score = scored['True Negative'].sum() + scored['True Positive'].sum()
      
        if(final_score > best_score):
            best_score = final_score
            best_thresh = i
            best_results = scored
            print("** New best score: " + str(best_score) + " with thresh: " + str(i))

    print("\n\nFinal best:")
    print("Best score: " + str(best_score) + " with thresh: " + str(best_thresh))
    flag_by_type(best_results)

    return best_thresh, best_score