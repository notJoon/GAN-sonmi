## renaming multiple files to numbers
## test folder : scrambled_file_name

## TODO : keep the files extension ...pathlib
"""
from pathlib import Path
p = Path('mysequence.fasta')
p.rename(p.with_suffix('.aln'))
"""

import os 

path = r'../scrambled_file_name'
files = os.listdir(path)

for idx, file in enumerate(files):
    os.rename(
        os.path.join(path, file), 
        os.path.join(path, ''.join([str(idx), '.txt']))
    )

print(f'{len(os.listdir(path))} of files has renamed')