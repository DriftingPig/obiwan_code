import multiprocessing
import sys
class CallGlobal(object):
    def __init__(self, pattern, globals, *args, **kwargs):
        self.pat = pattern
        self.args = args
        self.kwargs = kwargs
        self.globals = globals
    def getfunc(self, stage):
        func = self.pat % stage
        func = eval(func, self.globals)
        return func
    def getkwargs(self, stage, **kwargs):
        kwa = self.kwargs.copy()
        kwa.update(kwargs)
        return kwa
    def __call__(self, stage, **kwargs):
        func = self.getfunc(stage)
        kwa = self.getkwargs(stage, **kwargs)
        return func(*self.args, **kwa)

class CallGlobalTime(CallGlobal):
    def __call__(self, stage, **kwargs):
        from datetime import datetime
        print('Running stage', stage, 'at', datetime.now().isoformat())
        rtn = super(CallGlobalTime, self).__call__(stage, **kwargs)
        print('Stage', stage, 'finished:', datetime.now().isoformat())
        return rtn
    
class multiproc(object):
    
    def __init__(self, nthreads=1):
            if nthreads == 1 or nthreads is None:
                self.pool = None
            else:
                self.pool = multiprocessing.Pool(nthreads)
    def map(self, f, args):
        if self.pool:
            return self.pool.map(f, args)
        return list(map(f, args))

def flush(x=None):
    sys.stdout.flush()
    sys.stderr.flush()
    
    
def runstage(stage, stagefunc, prereqs={}, update=True, initial_args={}, **kwargs):

    # NOTE, if you add or change args here, be sure to update the recursive
    # "runstage" call below!!
    #NOTE: only kwargs that's put at the inital stage are the stuff that gets unchanged during all stages
    #kwargs is not changed all the way through (stable)
    #init args can be change all the time, as long as you return its value at the end of the stage
    #if something appears in both init args and kwargs, it is considered to be in kwargs
    #in general, you can never change outputs from previous runs, but you can add new variables. 
    #update P = update initial args, kwargs remain unchanged all the way through, so everytime a stage is finished, the init_args is changed, you can add or throw stuff you don't want here 
    print('Runstage', stage)

    try:
        prereq = prereqs[stage]
    except KeyError:
        prereq = stage - 1

    if prereq is None:
        P = initial_args
    else:
        P = runstage(prereq, stagefunc, prereqs=prereqs, update=update, initial_args=initial_args, **kwargs)
    #P.update(kwargs)
    Px = P.copy()
    Px.update(kwargs)

    print('Running stage', stage)
    # print('Prereq keys:', P.keys())
    # print('Adding kwargs keys:', kwargs.keys())
    # print('Combined keys:', Px.keys())

    R = stagefunc(stage, **Px)
    print('Stage', stage, 'finished')
    # if R is not None:
    #     print('Result keys:', R.keys())

    if update:
        if R is not None:
            P.update(R)
        R = P

    return R

