import os
import glob

from glass.__main__ import main


examples_dir = os.path.dirname(__file__)

print(f'examples directory: {examples_dir}')

examples = glob.glob(os.path.join(examples_dir, '*.cfg'))

print(f'found {len(examples)} examples')

for example in examples:
    name = os.path.basename(example).removesuffix('.cfg')
    print(f'running {name}...')

    main(['-q', example])
