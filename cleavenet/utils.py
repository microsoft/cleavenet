import os
import csv

originial_mmps = ['MMP1', 'MMP2', 'MMP3', 'MMP7', 'MMP8', 'MMP9', 'MMP10', 'MMP11',
        'MMP12', 'MMP13', 'MMP14', 'MMP15', 'MMP16', 'MMP17', 'MMP19', 'MMP20',
        'MMP24', 'MMP25']

mmps = ['MMP1', 'MMP10', 'MMP11', 'MMP12', 'MMP13', 'MMP14', 'MMP15', 'MMP16',
       'MMP17', 'MMP19', 'MMP2', 'MMP20', 'MMP24', 'MMP25', 'MMP3', 'MMP7',
       'MMP8', 'MMP9'] # the pivot table re-orderes MMPs in the creation of the train/test splits

bhatia_mmps = ['MMP1', 'MMP10', 'MMP12', 'MMP13', 'MMP17', 'MMP3', 'MMP7']

def get_data_dir():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_dir, "../data")

def get_save_dir():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_dir, "save")

def save_sequences(substrates, save_dir='outputs/'):
    save_file = os.path.join(save_dir, 'sequences.csv')
    os.makedirs(save_dir, exist_ok=True)
    with open(save_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(substrates)

def import_tensorflow():
    # Disable this function in __init__ if you dont know what you are doing!!
    # Filter tensorflow version warnings -
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf