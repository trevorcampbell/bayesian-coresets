import numpy as np
from ..base.iterative import IterativeCoreset
from ..riemann.kl import KLCoreset
from .. tangent import FixedFiniteTangentSpace

class IterativeHilbertCoreset(KLCoreset,IterativeCoreset):
    def __init__(self, N, tangent_space_factory, hilbert_coreset_class, step_sched = lambda i : np.sqrt(1./(1.+i)), optimize_iterations = True, num_its = 20):
        super().__init__(N=N) 
        self.tsf = tangent_space_factory
        self.step_sched = step_sched
        self.num_its = num_its
        self.hilbert_coreset_class = hilbert_coreset_class
        self.optimize_iterations = optimize_iterations

    def _stop(self):
        return self.itrs >= self.num_its #or self.error_so_far is below a threshold value #check for oboe?

    def _set_desired_coreset_size(self, M):
        self.M = M

    def _step(self):


        if(len(self.weights()[0]) == 0): #on first step, seed with a uniform subsample
            seeding_wts = (self.N/self.M) * np.random.multinomial(self.M, [1/self.N]*self.N)
            seeding_idcs = np.arange(self.N)[seeding_wts > 0]
            seeding_wts = seeding_wts[seeding_idcs]
            self._overwrite(seeding_idcs, seeding_wts) #I don't think it matters whether I call update or overwrite here?

        tangent_space = self.tsf(self.wts, self.idcs)


        scaling_vector = self.step_sched(self.itrs)*np.ones(tangent_space.num_vectors())
        scaling_vector[self.idcs] += ((1 - self.step_sched(self.itrs)) * self.wts)

        scaled_tangent_space= FixedFiniteTangentSpace(np.atleast_2d(scaling_vector).transpose() * tangent_space.vecs) #is there a better way to do this?

        hilbert_coreset_instantiated = self.hilbert_coreset_class(scaled_tangent_space)#self.tsf(self.wts, self.idcs))
        hilbert_coreset_instantiated.build(self.M)
        if self.optimize_iterations:
            try:
                hilbert_coreset_instantiated.optimize() 
            except RuntimeError:
                print("error optimizing!")#, ", matrix was ", hilbert_coreset_instantiated.T[hilbert_coreset_instantiated.idcs])
                #gained a small bit of accuracy just by terminating the algorithm here... since the case where nnls was terminating on iteration count was when our tangent space was basically zeroes. 
                #self.itrs = self.num_its
                #return
                #need to test eigenvalues here
                pass #need this line if I remove the print statement

        candidate_wts, candidate_idcs = hilbert_coreset_instantiated.weights() 

        candidate_wts *= scaling_vector[candidate_idcs]
        #self.itrs+= 1 is already done by iterative stuff that I'm inheriting from
        
        #add new wts
        self._overwrite(candidate_idcs, candidate_wts)
        if (self.size() > self.M):
            print("bug! coreset too big")

        #if (self.itrs %10 == 0):
        #    print("iteration: ", self.itrs)
            #print("step called, error for step was", hilbert_coreset_instantiated.error(), " , and whole coreset size is", self.size() , "with M = ", self.M, "\n") #can't see the full coreset's error: Trevor and I have to implement that later, in KLCoreset
        #do we want to use self.hilbert_coreset_instantiated.error() for something? deciding when to stop?

        #if (self. num_its - self.itrs < 3):
        #   print("iteration: ", self.itrs, " wts: ", self.wts)




#Silly things I've tried in reduce_coresets to address the issues with zero vectors in the tangent space:
    #num_negligible_weights = self.wts[self.wts < .0001].size
        #if (num_negligible_weights > 0):
        #    self.idcs = self.idcs[self.wts > 0.0001]
        #    self.wts = self.wts[self.wts > .0001]
        #    self.nwts -= num_negligible_weights
        #    self._refresh_views()
        #    if (self.nwts <= new_size):
        #        return
        
