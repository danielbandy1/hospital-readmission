"""Persist train-fitted category maps; never infer codes from one patient.
AUTHORED BY ORION
"""
import numpy as np

def fit_maps(frame):
    return {c: {value:i for i,value in enumerate(sorted(frame[c].astype('string').fillna('__MISSING__').unique()))}
            for c in frame.select_dtypes(['category','object','string']).columns}

def encode(frame,maps):
    frame=frame.copy()
    for c,mapping in maps.items():
        frame[c]=frame[c].astype('string').fillna('__MISSING__').map(mapping).fillna(-1).astype(float)
    return frame.astype(np.float32)
