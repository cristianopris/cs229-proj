import os
import random
import time

# The turibolt package is available on all Bolt tasks.
import turibolt as bolt

# The API lets you access the config file programmatically.
# This is useful for retrieving parameters or, as seen in more complex examples,
# constructing new configs for launching children.
#config = bolt.get_current_config()

variables = {}
execfile( "mnist_mlp.py", variables )

#bolt.set_status_message('Done')
print('I am done!')
