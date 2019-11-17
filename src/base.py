import os

join = os.path.join

# FILE EXTENSIONS
EXT_DATA = '.csv'
EXT_CKPT = '.ckpt' # pytorch checkpoint


# DIR NAMES
DNAME_DATA      = 'data'
DNAME_NOTEBOOKS = 'notebooks'
DNAME_SAVE      = 'save'
DNAME_MODELS    = 'models'
DNAME_SRC       = 'src'
DNAME_IMAGES    = 'images'

# FILE TREE
DIR_DATA      = join(DNAME_DATA)
DIR_NOTEBOOKS = join(DNAME_NOTEBOOKS)
DIR_SAVE      = join(DNAME_SAVE)
DIR_MODELS    = join(DNAME_MODELS)
DIR_SRC       = join(DNAME_SRC)
DIR_IMAGES    = join(DNAME_IMAGES)

def get_model_save_dir(model_name):
    return join(DIR_SAVE, model_name)

# FILE NAMES
CHECKPOINT_PREFIX = 'checkpoint'

# # FILE PATHS
# def get_model_save_path(model_name):
#     return join(get_model_save_dir(model_name), CHECKPOINT_PREFIX + EXT_CKPT)
