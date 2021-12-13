'''
adapted from astrometry.util.stage
this is a tutorial for me to understand what's going on with stage function, and it's not used anywhere. 
may delete it when I feel comfortable with using it. 
'''
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
        import pdb;pdb.set_trace()
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
    import pdb;pdb.set_trace()
    Px = P.copy()
    Px.update(kwargs)

    print('Running stage', stage)
    print(initial_args)
    print('Prereq keys:', P.keys())
    print('Adding kwargs keys:', kwargs.keys())
    print('Combined keys:', Px.keys())

    R = stagefunc(stage, **Px)
    print('Stage', stage, 'finished')
    # if R is not None:
    #     print('Result keys:', R.keys())

    if update:
        if R is not None:
            P.update(R)
        R = P

    return R

def stage_masks(init_duck=0,kwarg_duck=0, mp=None, **kwargs):
    print('stage: masks')
    print(init_duck,kwarg_duck)
    print(mp)
    init_duck=1
    kwarg_duck=1
    duck_mask = 1
    return dict(init_duck=init_duck, kwarg_duck=kwarg_duck,duck_mask=duck_mask)

def stage_brickstat(init_duck=0, kwarg_duck=0, **kwargs):
    print(init_duck,kwarg_duck)
    print('stage: brickstat')
    init_duck=2
    kwarg_duck=2
    print(kwargs)
    duck_mask = 100
    dup_duck = 'brickstat'
    return dict(init_duck=init_duck, kwarg_duck=kwarg_duck)

def stage_collect(init_duck=0, kwarg_duck=0, **kwargs):
    print(init_duck,kwarg_duck)
    print('stage: collect')
    init_duck=3
    kwarg_duck=3
    print(kwargs)
    return dict(init_duck=init_duck, kwarg_duck=kwarg_duck)

def stage_plots(init_duck=0,kwarg_duck=0, **kwargs):
    print(init_duck,kwarg_duck)
    print("stage: plots")
    init_duck=4
    kwarg_duck=4
    print(kwargs)
    return dict(init_duck=init_duck, kwarg_duck=kwarg_duck)

    
def post_run_brick(threads=None):
    
    kwargs = {}
    initargs = {}
    
    mp = multiproc(threads)
    kwargs.update(mp=mp)
    
    stagefunc = CallGlobalTime('stage_%s', globals())
    
    def mystagefunc(stage, mp=None, **kwargs):
        # Update the (pickled) survey output directory, so that running
        # with an updated --output-dir overrides the pickle file.
        flush()
        if mp is not None and threads is not None and threads > 1:
            # flush all workers too
            mp.map(flush, [[]] * threads)
        R = stagefunc(stage, mp=mp, **kwargs)
        flush()
        if mp is not None and threads is not None and threads > 1:
            mp.map(flush, [[]] * threads)
        return R
    
    prereqs = {
        'masks':None,
        'brickstat': 'masks',
        'collect': 'brickstat',
        'plots': 'collect'
        }
    
    kwargs.update(kwarg_duck=100,duck_trash=50,dup_duck=1)
    initargs.update(init_duck=100,duck_inittrash=20,dup_duck=1)
    R = None
    stages = ['plots']
    for stage in stages:
        R = runstage(stage, mystagefunc, prereqs=prereqs,
                     initial_args=initargs, **kwargs)
        
        

post_run_brick(threads=1)       
