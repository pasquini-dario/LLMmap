from .utility import *

def _inner_merge(A, B, inst):
    for system, system_data in B[inst].items():
        if not system in A[inst]:
            A[inst][system] = system_data
        else:
            A[inst][system].update(system_data)

def marge_data(A, B):
    for inst, inst_data in B.items():
        if not inst in A:
            A[inst] = B[inst]
            continue
        _inner_merge(A, B, inst)
        _inner_merge(B, A, inst)

def load_data(path, system_confs=None):
    logs = list(glob.glob(path))
    data = None
    for log in logs:
        print(f'Loading {log}')
        try:
            data_i = read_pickle(log).data
        except:
            print(f'Malformed: {log}')
            continue
        if system_confs:
            filter_system_confs(data_i, system_confs)
        if data is None:
            data = data_i
            
        marge_data(data, data_i)
        
    if data is None:
        data = {}
    return data