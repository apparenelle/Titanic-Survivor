import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from six.moves import urllib

# import tensorflow.v2.feature_column as fc

def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32): #A function whose literal job is to make functions
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function


if __name__ == "__main__":
    # Load Dataset.
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training data
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing data

    #print(dftrain.head)
    y_train = dftrain.pop('survived') #Removes the answers before we start training the ai
    y_eval = dfeval.pop('survived')

    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']

    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    # print(feature_columns)

    dftrain["embark_town"].unique()



    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)#is different because this is the test

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()
    print(result['accuracy'])



    result = list(linear_est.predict(eval_input_fn)) #This block displays results
    
    print(len(dfeval["sex"]))

    x=0
    counter = len(dfeval["sex"])
    while x < counter:
        print("\n\n\n")
        print(dfeval.loc[x])
        print(y_eval.loc[x])
        print(result[x]['probabilities'][1])
        x+=1