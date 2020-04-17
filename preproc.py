import os
from pathlib import Path
import numpy as np
import pandas as pd

data = pd.read_parquet('ml.ADRP-ADPR_pocket1_round1_dock.dsc.parquet')
data = data.sample(n=30000, random_state=0).reset_index(drop=True)
data = data.drop(columns=['cls','binner','name','smiles'])
data = data.iloc[:,:400]
data.to_parquet('./data.parquet')
print('Done.')

