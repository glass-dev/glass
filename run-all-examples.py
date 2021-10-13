import os
import glob
import time

from glass.__main__ import main


examples_dir = os.path.join(os.path.dirname(__file__), 'examples')

print(f'examples directory: {examples_dir}')

examples = glob.glob(os.path.join(examples_dir, '*.cfg'))

print(f'found {len(examples)} examples')

for example in examples:
    name = os.path.basename(example).removesuffix('.cfg')
    print(f'running {name}...')

    main(['-q', '--workdir', 'examples-output', example])

    # sleep one second to prevent output name clashes
    time.sleep(1)
