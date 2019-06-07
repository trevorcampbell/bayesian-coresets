TOL = 1e-16
from .tangent import FixedFiniteTangentSpace, MonteCarloFiniteTangentSpace
from .base import FullDataCoreset
from .hilbert import FrankWolfeCoreset, GIGACoreset, MatchingPursuitCoreset, ForwardStagewiseCoreset, OrthoPursuitCoreset, ImportanceSamplingCoreset, UniformSamplingCoreset, LassoCoreset
from .riemann import GreedyKLCoreset, GreedyQuadraticKLCoreset



