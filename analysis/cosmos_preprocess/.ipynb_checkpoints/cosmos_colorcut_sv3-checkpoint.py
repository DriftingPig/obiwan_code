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
def flux2mag(flux,mwtransmission):
        mag= 22.5 - 2.5 * np.log10(flux / mwtransmission)
        return mag

def get_scatter_ratio(equation, target_type='sv3_lrg'):
        #scatter info for cosmos 
        topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
        for n in range(10):
            dat = fits.getdata(topdir+'cosmos_set%d.fits'%n)
            #forgot to apply wisemask previously, do it here...
            sel = (dat['WISEMASK_W1']==0)
            dat = dat[sel]
            cut1_list = []
            ratio = []
            for i in range(10):
                if i==n:
                    continue
                sel_target = dat['set_%d_%s'%(n,target_type)]&dat['set_%d_matched'%i]
                zmag = flux2mag(dat[sel_target]['set_%d_flux_z'%i],dat[sel_target]['MW_TRANSMISSION_Z'])
                w1mag = flux2mag(dat[sel_target]['set_%d_flux_w1'%i],dat[sel_target]['MW_TRANSMISSION_W1'])
                rmag = flux2mag(dat[sel_target]['set_%d_flux_r'%i],dat[sel_target]['MW_TRANSMISSION_R'])
                gmag = flux2mag(dat[sel_target]['set_%d_flux_g'%i],dat[sel_target]['MW_TRANSMISSION_G'])
                zfibermag = flux2mag(dat[sel_target]['set_%d_fiberflux_z'%i],dat[sel_target]['MW_TRANSMISSION_Z'])
                cut1 = ~eval(equation)
                ratio.append(cut1.sum()/len(cut1))
            ave_ratio = np.mean(ratio)
            return ave_ratio

def get_cotamin_ratio(equation, equation_origin):
        #scatter info for cosmos 
        topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
        for n in range(10):
            dat = fits.getdata(topdir+'cosmos_set%d.fits'%n)
            #forgot to apply wisemask previously, do it here...
            sel = (dat['WISEMASK_W1']==0)
            dat = dat[sel]
            cut1_list = []
            ratio = []
            for i in range(10):
                if i==n:
                    continue
                sel_target = dat['set_%d_matched'%i]
                zmag = flux2mag(dat[sel_target]['set_%d_flux_z'%i],dat[sel_target]['MW_TRANSMISSION_Z'])
                w1mag = flux2mag(dat[sel_target]['set_%d_flux_w1'%i],dat[sel_target]['MW_TRANSMISSION_W1'])
                rmag = flux2mag(dat[sel_target]['set_%d_flux_r'%i],dat[sel_target]['MW_TRANSMISSION_R'])
                gmag = flux2mag(dat[sel_target]['set_%d_flux_g'%i],dat[sel_target]['MW_TRANSMISSION_G'])
                zfibermag = flux2mag(dat[sel_target]['set_%d_fiberflux_z'%i],dat[sel_target]['MW_TRANSMISSION_Z'])
                cut1 = eval(equation)
                cut2 = eval(equation_origin)
                ratio_i = (cut1&~cut2).sum()/cut2.sum()
                ratio.append(ratio_i)
            ave_ratio = np.mean(ratio)
            return ave_ratio        
        
def cosmos_colorcut(step=0.01, change=0.01):
    '''
    cut name should contain string _origin
    '''
    cut1_origin = 'zmag - w1mag > 0.8 * (rmag - zmag) - 0.6'
    cut2_origin = 'zfibermag < 21.7'
    #cut3 = cut3_1&cut3_2|cut3_3
    cut3_1_origin = 'gmag - rmag > 1.3'
    cut3_2_origin = '(gmag - rmag) > -1.55 * (rmag - w1mag)+3.13'
    cut3_3_origin = 'rmag - w1mag > 1.8'
    #cut4 = cut4_1&cut4_2|cut4_3
    cut4_1_origin = 'rmag - w1mag > (w1mag - 17.26) * 1.8'
    cut4_2_origin = 'rmag - w1mag > (w1mag - 16.36) * 1.'
    cut4_3_origin = 'rmag - w1mag > 3.29'
    
    #cut3_origin = '(%s)&(%s)|(%s)'%(cut3_1_origin,cut3_2_origin,cut3_3_origin)
    #cut4_origin = '(%s)&(%s)|(%s)'%(cut4_1_origin,cut4_2_origin,cut4_3_origin)
    
    equ_cut_all = "'(%s)&(%s)&((%s)&(%s)|(%s))&((%s)&(%s)|(%s))'%(cut1_origin, cut2_origin, cut3_1_origin, cut3_2_origin,cut3_3_origin,cut4_1_origin,cut4_2_origin,cut4_3_origin)"
    cut_all = eval(equ_cut_all)
    
    cuts =    ['cut1_origin','cut2_origin','cut3_1_origin','cut3_2_origin','cut3_3_origin','cut4_1_origin','cut4_2_origin','cut4_3_origin']
    symbols = ['-',   '+',   '-',     '-',     '-',      '-',     '-',     '-']
    
    #nothing need to be changed starting here
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
        ratio_old[i] = get_scatter_ratio(eval(cuts[i]))
        ratio_new[i] = ratio_old[i]
        cutlist[i] = [ratio_old[i]]
        l_cut[i] = [0]
        cut_N[i] = 0
    ratio_all_old = get_scatter_ratio(eval(equ_cutall_old))
    #all
    cutlist[len(cuts)] = [ratio_all_old] 
    #contamination
    cutlist[len(cuts)+1] = [0] 
        
    cutall_old = eval(equ_cutall_old)
    
    while(ratio_all_old>0.04):
        print(ratio_all_old)
        max_ratio = ratio_old.max()
        k = np.where(ratio_old==max_ratio)[0][0]
        while(ratio_old[k]-ratio_new[k]<change):
            cut_N[k]+=1
            exec(cut_old[k] + "=" + '"%s %s %f"'%(eval(cuts[k]), symbols[k], step*cut_N[k]))
            ratio_new[k] = get_scatter_ratio(eval(cut_old[k]))
        
        ratio_old[k] = ratio_new[k]
        ratio_cotamin = get_cotamin_ratio(eval(equ_cutall_old),cut_all)
        import pdb;pdb.set_trace()
        ratio_all_old = get_scatter_ratio(eval(equ_cutall_old))
        for i in range(len(cuts)):
            cutlist[i].append(ratio_old[i])
            l_cut[i].append(cut_N[i])
        cutlist[len(cuts)].append(ratio_all_old)
        cutlist[len(cuts)+1].append(ratio_cotamin)
        
    t = Table()
    for i in range(len(cuts)):
        cutname = cut_old[i].replace('_old','')
        t[cutname]=np.array(cutlist[i])
        t[cutname+'_n']=np.array(l_cut[i])
    t['cut_all'] = np.array(cutlist[len(cuts)])
    t['cut_cotamin'] = np.array(cutlist[len(cuts)+1])
    t.write('/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cuts_info_sv3.fits',overwrite=True)
    
cosmos_colorcut()    
