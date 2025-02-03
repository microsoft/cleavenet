import os

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
