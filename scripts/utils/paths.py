import os

DIR_PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_PATH_ARTIFACTS = os.path.join(DIR_PATH_ROOT, 'artifacts')
DIR_PATH_MODELS = os.path.join(DIR_PATH_ARTIFACTS, 'models')
DIR_PATH_DATA = os.path.join(DIR_PATH_ARTIFACTS, 'data')
DIR_PATH_PLOTS = os.path.join(DIR_PATH_ARTIFACTS, 'plots')
DIR_PATH_LOGS = os.path.join(DIR_PATH_ARTIFACTS, 'logs')

DATA_PATH_ERROR_X_INIT_IN = os.path.join(DIR_PATH_DATA, 'error_x_init_in.npz')
DATA_PATH_ERROR_X_INIT_OUT = os.path.join(DIR_PATH_DATA, 'error_x_init_out.npz')
DATA_PATH_ERROR_X_INIT_IN_NOISE = os.path.join(DIR_PATH_DATA, 'error_x_init_in_noise.npz')
DATA_PATH_ERROR_PROFILE_META_ADAPTED_X_INIT_IN = (
    os.path.join(DIR_PATH_DATA, 'error_profile_meta_adapted_x_init_in.npz'))
DATA_PATH_ERROR_PROFILE_META_ADAPTED_X_INIT_OUT = (
    os.path.join(DIR_PATH_DATA, 'error_profile_meta_adapted_x_init_out.npz'))
DATA_PATH_ERROR_META_ADAPTED_X_INIT_SAMPLING = (
    os.path.join(DIR_PATH_DATA, 'error_meta_adapted_x_init_sampling.npz'))
DATA_PATH_STATE_META_ADAPTED_X_INIT = os.path.join(DIR_PATH_DATA, 'state_meta_adapted_x_init.npz')
DATA_PATH_STATE_META_ADAPTED_X_INIT_NOISE = os.path.join(DIR_PATH_DATA, 'state_meta_adapted_x_init_noise.npz')

DATA_PATH_PARAM_ERROR_LMBDA = os.path.join(DIR_PATH_DATA, 'param_error_lmbda.npz')
DATA_PATH_PARAM_ERROR_LMBDA_PARALLEL = os.path.join(DIR_PATH_DATA, 'param_error_lmbda_parallel.npz')
