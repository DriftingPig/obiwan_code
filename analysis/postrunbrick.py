'''
adapted from astrometry.util.stage

quick example to call it: 
# python postrunbrick.py --survey-dir /global/project/projectdirs/cosmo/data/legacysurvey/dr9/ --outdir /global/cfs/cdirs/desi/users/huikong/decals_ngc/production_run/ --threads 30 --dataset normal --nobj 200 --startid 0

survey-dir: set to the current data release directory
outdir: output directory
dataset: select between normal/cosmos
nobj: number of objects injected
startid: startid of the randoms 


structure of output directory:
in: /global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/, with run name deep1:
--brickstat/
  --brickstat.py
  --deep1/
    --FinishedBricks.txt (finished bricks)
    --UnfinishedBricks.txt  (unfinished brick)
  --real_bricklists/
    --bricks_deep1.txt
--deep1/
  bricklist.txt  
  divided_randoms/brick_1480m035.fits...  
  output/  
  randoms/  
  seed.fits  
  subset/
  

#TODO!!!
add a 'tracer' option
'''

from utils import CallGlobalTime, multiproc, flush, runstage
from cosmos import CosmosRepeats
from SurveySource import BaseSource
def stage_cosmos(mp=None, **kwargs):
    '''
    make masks for cosmos repeats
    '''
    

    survey_dir = kwargs.get('surveydir')
    outdir = kwargs.get('outdir')
    catalog = CosmosRepeats(survey_dir=survey_dir, outdir=outdir,subsection='cosmos80',startid = kwargs.get('startid'))
    catalog.set_maskbits_cross(mp=mp)
    print('maskbits_cross complete')
    catalog.mask_tractor(mp=mp)
    print('mask tractor complete')
    catalog.match_catalog(mp=mp)
    print('match_catalog complete')
    return dict(bricklist=catalog.bricklist)

def stage_brickstat(mp=None, **kwargs):
    """
    the only purpose here is to return a list of bricks to be processed. change it any way you want!
    """
    print('stage: brickstat')
    if kwargs['dataset'] == 'cosmos':
        survey_dir = kwargs.get('surveydir')
        outdir = kwargs.get('outdir')
        catalog = CosmosRepeats(survey_dir=survey_dir, outdir=outdir,subsection='cosmos80',startid = kwargs.get('startid'))
        return dict(bricklist=catalog.bricklist)
    else:
        import os
        import numpy as np
        outdir = kwargs.get('outdir')
        topdir = os.path.dirname(os.path.abspath(outdir))
        name = os.path.basename(topdir)
        topdir = os.path.dirname(os.path.abspath(topdir))
        topdir = os.path.join(topdir,'brickstat',name)
        
        bricklist = np.loadtxt(topdir+'/FinishedBricks.txt',dtype=np.str)
        if kwargs['totslice']>1:
            i = kwargs['sliceidx']
            n = kwargs['totslice']
            bricklist = bricklist[i::n]
        return dict(bricklist=bricklist)

def stage_collect(mp=None, **kwargs):
    print('stage: collect')
    if kwargs['dataset'] == 'cosmos':
        mode = 'tractor'
        survey_dir = kwargs.get('surveydir')
        outdir = kwargs.get('outdir')
        startid = kwargs.get('startid')
        cms = CosmosRepeats(survey_dir=survey_dir, outdir=outdir,subsection='rs%d_cosmos80'%startid,startid = kwargs.get('startid'))
        total_sets = cms.total_sets
        bricklist = cms.bricklist
        from collect import Collect
        clt = Collect(survey_dir=survey_dir, outdir=outdir,subsection='rs%d_cosmos80'%startid)
        for one_set in total_sets:
            clt.brick_match(threads = kwargs['threads'], bricklist = bricklist, mp=mp, subsection='rs%d_cosmos%s'%(startid,one_set), startid=kwargs['startid'], nobj=kwargs['nobj'],mode = mode, tracer = kwargs['tracer'])
    else:
        mode = 'sim'
        region = 'south'
        survey_dir = kwargs.get('surveydir')
        outdir = kwargs.get('outdir')
        subsection = kwargs.get('subsection')
        bricklist = kwargs['bricklist']
        from collect import Collect
        clt = Collect(survey_dir=survey_dir, outdir=outdir,subsection=subsection,region= region)
        clt.brick_match(threads = kwargs['threads'], bricklist = bricklist, mp=mp, subsection=subsection, startid=kwargs['startid'], nobj=kwargs['nobj'], mode = mode, tracer = kwargs['tracer'], totslice = kwargs['totslice'], sliceidx = kwargs['sliceidx'])
        clt.compute_photoz(totslice = kwargs['totslice'], sliceidx = kwargs['sliceidx'])
        #clt.brick_match_dr9(bricklist = bricklist,tracer = kwargs['tracer'],threads = kwargs['threads'], mp=mp, totslice = kwargs['totslice'], sliceidx = kwargs['sliceidx'])
        #clt.brick_match_random(bricklist = bricklist,threads = kwargs['threads'], mp=mp, totslice = kwargs['totslice'], sliceidx = kwargs['sliceidx'])
        #clt.brick_match_input_random(bricklist = bricklist,threads = kwargs['threads'], mp=mp)
        #clt.collect_sweep( south = True,threads = kwargs['threads'], mp=mp,tracer = kwargs['tracer'])
    return None

def stage_plots(mp=None, **kwargs):
    print("stage: plots")
    import sys
    sys.path.append('./baseline_plots')
    from fig1 import fig1, fig1b,fig1_cosmos,fig1b_cosmos,fig1c,fig1d,fig1e_cosmos,fig1e_cosmos_ks
    from fig2 import fig2
    from systematic import sys_from_map,make_sys_plot,density_plot
    startid = kwargs.get('startid')
    if kwargs['dataset'] == 'cosmos':
        cms = CosmosRepeats(survey_dir=kwargs['surveydir'], outdir=kwargs['outdir'],subsection='rs%d_cosmos80'%startid, startid = kwargs.get('startid'))
        total_sets = cms.total_sets
        for one_set in total_sets:
            catalog = BaseSource(survey_dir=kwargs['surveydir'], outdir=kwargs['outdir'],subsection = 'rs%d_cosmos%s'%(startid,one_set))
            catalog.get_processed_one()
            fig1(catalog,startid)
            fig1b(catalog,startid)
            fig1_cosmos(catalog,startid)
            fig1b_cosmos(catalog,'g',one_set,'lrg_sv3',startid)
            fig1b_cosmos(catalog,'r',one_set,'lrg_sv3',startid)
            fig1b_cosmos(catalog,'z',one_set,'lrg_sv3',startid)
            fig1b_cosmos(catalog,'w1',one_set,'lrg_sv3',startid)
            fig1c(catalog,startid)
            fig1d(catalog,startid)
            fig1e_cosmos(catalog, one_set, startid)
            fig1e_cosmos_ks(catalog, one_set, startid)
            fig2(catalog,startid)
            #sys_maps(catalog,sys = 'ebv', percentile=1,startid=startid)
    if kwargs['dataset'] == 'normal':
        catalog = BaseSource(survey_dir=kwargs['surveydir'], outdir=kwargs['outdir'],subsection = 'rs%d'%(startid))
        catalog.get_processed_one()
        fig1(catalog,startid)
        fig1b(catalog,startid)
        fig1c(catalog,startid)
        fig1d(catalog,startid)
        fig2(catalog,startid)
        #make_sys_plot(catalog,startid)
        density_plot(catalog,startid)
    return None


    
def stage_stack_rs(mp=None, **kwargs):
    survey_dir = kwargs.get('surveydir')
    outdir = kwargs.get('outdir')
    if kwargs['dataset'] == 'cosmos':
        catalog = CosmosRepeats(survey_dir=survey_dir, outdir=outdir,subsection='cosmos80',startid = kwargs.get('startid'))
        catalog.stack_all_rs()
    else:
        catalog = BaseSource(survey_dir=survey_dir, outdir=outdir,subsection='cosmos80',startid = kwargs.get('startid'))
        startids = ['0','50','100', '150','200','250','350']
        catalog.stack_all_rs(startids=startids)
    
def get_parser():
    import argparse
    de = ('Hi' +
          'What should I write here?')

    ep = '''
    tutorial on how to use this code
    '''
    parser = argparse.ArgumentParser(description=de,epilog=ep)
    
    parser.add_argument('-d', '--outdir', dest='output_dir', required = True,
                        help="the $obiwan_out directory")
    parser.add_argument('--survey-dir', type=str, default=None, required = True,
                        help='the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--threads', type=int, required = True, help='Run multi-threaded')
    parser.add_argument('--dataset', type=str, default = 'normal', choices=['normal','cosmos'], help='the dataset to process')
    parser.add_argument('--startid', type=int, default=0, help='starting point of injection')
    parser.add_argument('--nobj', type=int, default=0, required = True, help='total number of sources injected')
    parser.add_argument('--tracer', type=str, default = 'lrg_sv3', choices=['lrg_sv3','elg_sv3','LRG'], help='the tracer we are interested in')
    parser.add_argument('--subsection', type=str, default='none')
    parser.add_argument('--tot-slice', type=int, default=1)
    parser.add_argument('--slice-idx', type=int, default=0)
    return parser
    
def post_run_brick(args=None):
    parser = get_parser()
    opt = parser.parse_args(args=args)
    optdict = vars(opt)
    surveydir = optdict.pop('survey_dir', None)
    outdir   = optdict.pop('output_dir', 0)
    threads = optdict.pop('threads')
    dataset = optdict.pop('dataset')
    startid = optdict.pop('startid')
    nobj = optdict.pop('nobj')
    tracer = optdict.pop('tracer')
    subsection = optdict.pop('subsection')
    tot_slice = optdict.pop('tot_slice')
    slice_idx = optdict.pop('slice_idx')
    
    kwargs = {}
    initargs = {}
    
    mp = multiproc(threads)
    kwargs.update(mp=mp)
    kwargs.update(surveydir=surveydir)
    kwargs.update(outdir=outdir)
    kwargs.update(dataset=dataset)
    kwargs.update(startid=startid)
    kwargs.update(nobj=nobj)
    kwargs.update(threads=threads)
    kwargs.update(totslice=tot_slice)
    kwargs.update(sliceidx=slice_idx)
    if subsection == 'none':
        kwargs.update(subsection="rs%d"%startid)
    else:
        kwargs.update(subsection=subsection)
    kwargs.update(tracer=tracer)
    
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
    
    if dataset == 'cosmos':
        prereqs = {
        'cosmos':None,
        'collect': 'cosmos',
        'plots': None,
        'brickstat':None,
        'stack_rs':None
        }
    elif dataset == 'normal':
        prereqs = {
        'brickstat':None,
        'collect': 'brickstat',
        'plots': None,
        'stack_rs':None
        }
    
    
    R = None
    stages = ['collect']
    for stage in stages:
        R = runstage(stage, mystagefunc, prereqs=prereqs,
                     initial_args=initargs, **kwargs)
        
        

if __name__ == '__main__':
    post_run_brick()
    

