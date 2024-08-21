import sys
sys.path.append('imports')
from generate_training import generate_data_multithreaded
import dill
from socket import gethostname

if gethostname() == 'Louis_PC':
    param_path = r'C:\Users\Pouis\Documents\Uni Shit\Masters\PhaseGit\Supervised-clustering-phase-pBEC\MultiMode Analysis\training_params.dill'

with open(param_path, 'rb') as f:
    kwargs = dill.load(f)

generate_data_multithreaded(**kwargs)
