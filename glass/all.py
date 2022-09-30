# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''meta-module that imports all modules from the GLASS namespace'''


(lambda: [__import__(_.name) for _ in __import__('pkgutil').iter_modules(__import__(__package__).__path__, __package__ + '.')])()  # noqa
