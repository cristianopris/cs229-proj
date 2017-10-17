import sys
print('\n'.join(sys.path))
from mpi4py import MPI

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector

from gym.spaces import Box

Box(np.array([0,0], np.array([359,359])))