from math import *
from numpy import *
import numpy as np
from matplotlib.cbook import flatten

class SSA_Stim:
    def __init__(self, SSAType, Standard, Deviant, StandPro, DeviPro, Presentations):
        self.SSAType        = SSAType
        self.Standard       = Standard
        self.Deviant        = Deviant
        self.StandPro       = StandPro
        self.DeviPro        = DeviPro
        self.Presentations  = Presentations

    def NoSSA(seed):
        # Random tones on a scale (equal probability for each tone)
        jumps = 0.2
        AudStims = list(2**(arange(0, 4+jumps, jumps) + log2(2000))) * 20
        print('%i tone presentations'%len(AudStims))
        print('******')
        
        rndd = np.random.RandomState(seed)
        rndd.shuffle(AudStims)
        rndd.shuffle(AudStims)
        rndd.shuffle(AudStims)
        rndd.shuffle(AudStims)

        return AudStims

    def TwoStims(self):
        AudStims = [self.Standard]*int(self.StandPro*self.Presentations) + [self.Deviant]*int(self.DeviPro*self.Presentations)
        
        return AudStims

    def DeviantAlone(self):
        AudStims = [self.Standard]*int(self.DeviPro*self.Presentations) + [self.Deviant]*int(self.StandPro*self.Presentations)

        return AudStims

    def TwoStimsPeriodic(self):
        standard_per_period = int(round(self.StandPro/self.DeviPro))
        n_periods = int(round(self.Presentations*self.DeviPro))

        AudStims = list(flatten([[self.Standard]*standard_per_period + [self.Deviant] for t in range(n_periods)]))

        return AudStims

    def TwoStimsPeriodicP9_29(self):
        AudStims= list(flatten([self.Deviant] + [[self.Standard]*9 + [self.Deviant] + [self.Standard]*29 + [self.Deviant] for i in range(12)] + [self.Standard]*19))

        return AudStims

    def DiverseNarrow_T_or_Exp(self, d_octave, n_extend):  
        # T: d_octave = 5, n_extend = 2
        # Exp: d_octave = 11, n_extend = 4
        if self.Deviant<self.Standard: raise Exception("Deviant should be bigger than Standard, just a def")

        AudStims = []
        
        df = (log2(self.Deviant) - log2(self.Standard))/d_octave # Number (fraction) of octaves between consecutive frequencies
        
        LowestFreq = log2(self.Standard) - n_extend*df
        
        DiverseNarrow = [int(round(2**(LowestFreq + i*df))) for i in range(d_octave+(n_extend*2)+1)]
        
        frac_repetitions = 1 / len(DiverseNarrow)

        [AudStims.extend([freq]*int(frac_repetitions*self.Presentations)) for freq in DiverseNarrow]

        return AudStims

    def DiverseBroad(self, DiverseBroadProbs, SSAType, n_extend=2, d_octave=1):
        # Same as narrow, different probabilities (nonuniform) and parameters (I'm too lazy to merge)
        # 1 rounding difference between Oren's original and this; I guess it doesn't matter and this is more readable
        if Deviant<Standard: raise Exception("Deviant should be bigger than Standard, just a def")
        if SSAType == 'DiverseBroadExp': 
            n_extend = 5
            d_octave = 1
            raise Exception('This will not work until I have a larger column')

        AudStims = []
        
        df = ((log2(self.Deviant) - log2(self.Standard))) / d_octave
        
        LowestFreq = log2(self.Standard) - n_extend*df
        
        DiverseBroad = [int(round(2**(LowestFreq + i*df))) for i in range(d_octave + (n_extend*2) + 1)]

        # Each stimulus appears according to its probability
        [AudStims.extend([freq]*int(DiverseBroadProbs[s]*self.Presentations)) for s,freq in enumerate(DiverseBroad)]

        return AudStims


