import subprocess
from glob import glob
import astropy.io.fits as fits
from astropy.table import Table
import os
import numpy as np
'''
currently considering LRG sv3 selection
https://github.com/desihub/desitarget/blob/master/py/desitarget/sv3/sv3_cuts.py

cosmos_colorcut.py is a more readable version, while this is a more adaptable version
'''
class scatter_ratio(object):
    def __init__(self,target_type='sv3_elg',cotamin = False):
        topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
        keys = ['zmag','w1mag','rmag','zfibermag','gmag','gfibermag']
        band = {'zmag':'z','w1mag':'w1','rmag':'r','zfibermag':'z','gmag':'g','gfibermag':'g'}
        for key in keys:
            exec(key + '= [[None for _ in range(10)] for _ in range(10)]')
        for n in range(10):
            dat = fits.getdata(topdir+'cosmos_set%d.fits'%n)
            for i in range(10):
                if i == n:
                    continue
                if cotamin:
                    sel_target = dat['set_%d_matched'%i]
                else:
                    sel_target = dat['set_%d_%s'%(n,target_type)]&dat['set_%d_matched'%i]
                for key in keys:
                    if key == 'zfibermag' or key == 'gfibermag' or key == 'rfibermag':
                        exec(key+'[n][i] = self.flux2mag(dat[sel_target]['+'"set_%d_fiberflux_%s"'%(i,band[key])+'],dat[sel_target]["MW_TRANSMISSION_%s"])'%band[key].upper())
                    else:
                        exec(key+'[n][i] = self.flux2mag(dat[sel_target]['+'"set_%d_flux_%s"'%(i,band[key])+'],dat[sel_target]["MW_TRANSMISSION_%s"])'%band[key].upper())
        R = locals()
        for key in keys:
            setattr(self,key,R[key])
        
    @staticmethod            
    def flux2mag(flux,mwtransmission):
        mag= 22.5 - 2.5 * np.log10(flux / mwtransmission)
        return mag

    def get_scatter_ratio(self,equation):
        for n in range(10):
            cut1_list = []
            ratio = []
            for i in range(10):
                if i==n:
                    continue
                zmag = self.zmag[n][i]
                w1mag = self.w1mag[n][i]
                rmag = self.rmag[n][i]
                gmag = self.gmag[n][i]
                zfibermag = self.zfibermag[n][i]
                gfibermag = self.gfibermag[n][i]
                cut1 = ~eval(equation)
                ratio.append(cut1.sum()/len(cut1))
            ave_ratio = np.mean(ratio)
            return ave_ratio

    def get_cotamin_ratio(self, equation, equation_origin):
        for n in range(10):
            cut1_list = []
            ratio = []
            for i in range(10):
                if i==n:
                    continue
                zmag = self.zmag[n][i]
                w1mag = self.w1mag[n][i]
                rmag = self.rmag[n][i]
                gmag = self.gmag[n][i]
                zfibermag = self.zfibermag[n][i]
                gfibermag = self.gfibermag[n][i]
                cut1 = eval(equation)
                cut2 = eval(equation_origin)
                ratio_i = (cut1&~cut2).sum()/cut2.sum()
                ratio.append(ratio_i)
            ave_ratio = np.mean(ratio)
            return ave_ratio        
        
def cosmos_colorcut():
    '''
    cut name should contain string _origin
    '''
    from cuts_record import cuts_record
    cr = cuts_record('ELG_sv3')
    frozen_keys = ['equ_cut_all','cut_all','cuts','symbols','step','change','lower_limit']
    for key in cr.keys:
        if key in frozen_keys:
            continue
        exec("global %s; "%key+key+'='+"'%s'"%getattr(cr,key))
    for key in ['equ_cut_all','cut_all']:
        #for strings
        exec("global %s; "%key+key+'='+'"%s"'%getattr(cr,key))
    for key in ['cuts','symbols','step','change','lower_limit']:
        #for non strings
        exec("global %s; "%key+key+'='+'%s'%getattr(cr,key))

    cls_sr = scatter_ratio(cotamin=False,target_type = cr.target_type)
    cls_sr_cotamin = scatter_ratio(cotamin=True, target_type = cr.target_type)
    ratio_old = np.zeros(len(cuts))
    ratio_new = np.zeros(len(cuts))
    cut_old = cuts.copy()
    equ_cutall_old = equ_cut_all
    equ_cutall_old = equ_cutall_old.replace('origin','old')
    cutlist = [None]*(len(cuts)+2)
    #step for each cut
    cut_N = [None]*len(cuts)
    l_cut = [None]*len(cuts)
    for i in range(len(cuts)):
        cut_old[i] = cut_old[i].replace('origin','old')
        exec(cut_old[i]+'="%s"'%eval(cuts[i]))
        ratio_old[i] = cls_sr.get_scatter_ratio(eval(cuts[i]))
        ratio_new[i] = ratio_old[i]
        cutlist[i] = [ratio_old[i]]
        l_cut[i] = [0]
        cut_N[i] = 0
    ratio_all_old = cls_sr.get_scatter_ratio(eval(equ_cutall_old))
    #all
    cutlist[len(cuts)] = [ratio_all_old] 
    #contamination
    cutlist[len(cuts)+1] = [0] 
        
    cutall_old = eval(equ_cutall_old)
    
    while(ratio_all_old>lower_limit):
        print(ratio_all_old)
        max_ratio = ratio_old.max()
        k = np.where(ratio_old==max_ratio)[0][0]
        while(ratio_old[k]-ratio_new[k]<change):
            cut_N[k]+=1
            exec(cut_old[k] + "=" + '"%s %s %f"'%(eval(cuts[k]), symbols[k], step*cut_N[k]))
            ratio_new[k] = cls_sr.get_scatter_ratio(eval(cut_old[k]))
        
        ratio_old[k] = ratio_new[k]
        ratio_cotamin = cls_sr_cotamin.get_cotamin_ratio(eval(equ_cutall_old),cut_all)
        ratio_all_old = cls_sr.get_scatter_ratio(eval(equ_cutall_old))
        for i in range(len(cuts)):
            cutlist[i].append(ratio_old[i])
            l_cut[i].append(cut_N[i])
        cutlist[len(cuts)].append(ratio_all_old)
        cutlist[len(cuts)+1].append(ratio_cotamin)
        print(ratio_old)
    t = Table()
    for i in range(len(cuts)):
        cutname = cut_old[i].replace('_old','')
        t[cutname]=np.array(cutlist[i])
        t[cutname+'_n']=np.array(l_cut[i])
    t['cut_all'] = np.array(cutlist[len(cuts)])
    t['cut_cotamin'] = np.array(cutlist[len(cuts)+1])
    t.write(savedir,overwrite=True)
    print(savedir)
    
cosmos_colorcut()    
