#the delta flux vs stuff
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import matplotlib.mlab as mlab
import matplotlib
from scipy.stats import norm


def one_sigma_2dplot(y_value,x_value, title_x=None, title_y=None, x_boundary_percentile=1, percentile_y=16, y_boundary=5,bins=40,y_boundary_percentile=5,write_mean=False, write_mean_fn = None):
		'''
		2d density plot with 3 percentile line drew on the plot
		'''
		sel = (x_value>=-10e9)&(x_value<=10e9)&(y_value>=-10e9)&(y_value<=10e9)
		x_value = x_value[sel]
		y_value = y_value[sel]
		lower = np.percentile(x_value,x_boundary_percentile)
		higher = np.percentile(x_value,100-x_boundary_percentile)
		lower_y = np.percentile(y_value,y_boundary_percentile)
		higher_y = np.percentile(y_value,100-y_boundary_percentile)    
		sel2 = (x_value>lower)&(x_value<higher)
        
		y = y_value[sel2]
		x = x_value[sel2]
    
		plt.xlabel(title_x)
		plt.ylabel(title_y)
		yl,xl = _get_percentile(y,x,bins,100-percentile_y)
		plt.plot(xl,yl,color = 'y')
		yl,xl = _get_percentile(y,x,bins,50)
		yl_mean = yl;xl_mean=xl
		plt.plot(xl,yl,color = 'y')
		yl,xl = _get_percentile(y,x,bins,percentile_y)
		plt.plot(xl,yl,color = 'y')
		plt.plot(xl,[0]*len(xl),'r:',alpha=0.5)
		if write_mean:#write mean line to a file 
			assert(write_mean_fn is not None)
			dat = np.array([xl_mean,yl_mean]).transpose()
			np.savetxt(write_mean_fn,dat)

        
		if y_boundary is not None:
			sel3 = (y_value>lower_y)&(y_value<higher_y)
			y2 = y_value[sel3]
			x2 = x_value[sel3]
		else:
			y2 = y_value
			x2 = x_value

		h,xe,ye,i = plt.hist2d(x2,y2,bins=40,cmap = 'Blues')
		plt.plot(x2,[0]*len(x2),'r:',alpha=0.5)

		plt.gca().set_xlim((x2.min(),x2.max()))
		plt.gca().set_ylim((y2.min(),y2.max()))


def _get_percentile(y,x,bins,percen_num):#16,50,84
		minimum = x.min()
		maximum = x.max()
		interval = (maximum-minimum)/bins
		percent_list=[]
		mid_list = []
		for i in range(bins):
			left = minimum+i*interval
			right = minimum+(i+1)*interval
			mid = minimum+(i+0.5)*interval
			y_i = y[(x>left)&(x<=right)]
			if len(y_i)>0:
				output = np.percentile(y_i,percen_num)
				percent_list.append(output)
				mid_list.append(mid)
		return percent_list,mid_list


def one_subplot_hist(title=None,variable=None,xlabel = None,ylabel=None,percentile=5,bins=50,density=False, mean = False,color='b',label=None,text=False,cdf = False):
		sel = (variable>=-10e9)&(variable<=10e9)
		variable = variable[sel]
		lower = np.percentile(variable,percentile)
		higher = np.percentile(variable,100-percentile)
		sel2 = (variable>=lower)&(variable<=higher)
		variable = variable[sel2]
		vmean = variable.mean()
		if mean:
			variable = variable - variable.mean()
		if mean:
			n,bins,_ = plt.hist(variable,bins=bins,density=density, histtype = 'step', linewidth=2, color=color,label = "mean: %.3f"%vmean,cumulative=cdf)
		else:
			n,bins,_ = plt.hist(variable,bins=bins,density=density, histtype = 'step', linewidth=2, color=color,label=label,cumulative=cdf)
		y1_cen = n.max()*3/4
		y2_cen = n.max()*5/8
		x_cen = bins[int(len(bins)*5/8)]
		if text is True:
			plt.text(x_cen,y1_cen,"mean:%.3f"%vmean)
			plt.text(x_cen,y2_cen,"std:%.3f"%variable.std())
		if xlabel is not None:
			plt.xlabel(xlabel)
		if ylabel is not None:
			plt.ylabel(ylabel)
		if title is not None:
			plt.title(title)
		if label is not None:
			plt.legend()        
		return variable, n.max(), n, bins 

def flux_plot_sigma(x_value,f_type):
    
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(x_value), xlabel = r'$\Delta$ %s flux (output - input)*sqrt(ivar)'%f_type, ylabel = 'PDF', bins=30, density = True, mean= True)

    (mu, sigma) = norm.fit(delta_flux)
    d = mlab.normpdf( bins, 0, sigma)
    l = plt.plot(bins, d, 'k-', linewidth=1,label = 'fit '+r'$\sigma$ = %.2f'%sigma)
    d = mlab.normpdf( bins, 0, 1)
    l = plt.plot(bins, d, 'k--', linewidth=1,label = 'Standard Norm')
    plt.plot([0]*2,[0,max_bin],'k:')
    plt.gca().set_ylim((0,max_bin*1.03))
    plt.legend(loc = 'upper left', prop={'size': 9})          
            
def fig1(catalog,startid):
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
        
    source = catalog.processed_one
    sel = source['matched']&(source['maskbits']==0)
    source = source[sel]
    g_truth = 22.5 - 2.5 * np.log10(source['sim_gflux'] / source['mw_transmission_g'])
    r_truth = 22.5 - 2.5 * np.log10(source['sim_rflux'] / source['mw_transmission_r'])
    z_truth = 22.5 - 2.5 * np.log10(source['sim_zflux'] / source['mw_transmission_z'])
    w1_truth = source['sim_w1']
    #w2_truth = source['sim_w2']
    g_measure = 22.5 - 2.5 * np.log10(source['flux_g'] / source['mw_transmission_g'])
    r_measure = 22.5 - 2.5 * np.log10(source['flux_r'] / source['mw_transmission_r'])
    z_measure = 22.5 - 2.5 * np.log10(source['flux_z'] / source['mw_transmission_z'])
    w1_measure = 22.5 - 2.5 * np.log10(source['flux_w1'] / source['mw_transmission_w1'])
    w2_measure = 22.5 - 2.5 * np.log10(source['flux_w2'] / source['mw_transmission_w2'])
    
    gflux_truth = source['sim_gflux']
    rflux_truth = source['sim_rflux']
    zflux_truth = source['sim_zflux']
    w1flux_truth = 10**((22.5 - source['sim_w1'])/2.5)*source['mw_transmission_w1']
    #w2flux_truth = 10**((22.5 - source['sim_w2'])/2.5)*source['mw_transmission_w2']
    
    gflux_measure = source['flux_g']
    rflux_measure = source['flux_r']
    zflux_measure = source['flux_z']
    w1flux_measure = source['flux_w1']
    #w2flux_measure = source['flux_w2']
    
    plt.figure(figsize=(9,18))  
    plt.subplot(5,2,1)    
    one_sigma_2dplot(g_truth-g_measure,g_measure,title_x = 'g true',title_y = r'$\Delta$ g (input-output)')
    plt.subplot(5,2,3)    
    one_sigma_2dplot(r_truth-r_measure,r_measure,title_x = 'r true',title_y = r'$\Delta$ r (input-output)')
    plt.subplot(5,2,5)    
    one_sigma_2dplot(z_truth-z_measure,z_measure,title_x = 'z true',title_y = r'$\Delta$ z (input-output)')
    plt.subplot(5,2,7)
    one_sigma_2dplot(w1_truth-w1_measure,w1_measure,title_x = 'w1 true',title_y = r'$\Delta$ w1 (input-output)')
    #plt.subplot(5,2,8)
    #one_sigma_2dplot(w2_truth-w2_measure,w2_measure,title_x = 'w2 true',title_y = r'$\Delta$ w2 (input-output)')
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    plt.subplot(5,2,2)
    #,write_mean=False, write_mean_fn = None
    fn = topdir+"/flux_diff_g.txt"
    one_sigma_2dplot(gflux_measure - gflux_truth, g_measure, title_x = 'g true',title_y = r'$\Delta$ g flux (output-input)',write_mean=True,write_mean_fn=fn)
    plt.subplot(5,2,4)
    fn = topdir+"/flux_diff_r.txt"
    one_sigma_2dplot(rflux_measure - rflux_truth, r_measure, title_x = 'r true',title_y = r'$\Delta$ r flux (output-input)',write_mean=True,write_mean_fn=fn)
    plt.subplot(5,2,6)
    fn = topdir+"/flux_diff_z.txt"
    one_sigma_2dplot(zflux_measure - zflux_truth, z_measure, title_x = 'z true',title_y = r'$\Delta$ z flux (output-input)',write_mean=True,write_mean_fn=fn)
    plt.subplot(5,2,8)
    fn = topdir+"/flux_diff_w1.txt"
    one_sigma_2dplot(w1flux_measure - w1flux_truth, w1_measure, title_x = 'w1 true',title_y = r'$\Delta$ w1 flux (output-input)',write_mean=True,write_mean_fn=fn)
    #plt.subplot(5,2,10)
    #one_sigma_2dplot(w2flux_measure - w2flux_truth, w2_measure, title_x = 'delta w2 flux',title_y = r'$\Delta$ w2 (output-input)')
    
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1.png')
    
    
def fig1b(catalog,startid):
    source = catalog.processed_one
    sel = source['matched']&(source['maskbits']==0)
    source = source[sel]
    g_truth = 22.5 - 2.5 * np.log10(source['sim_gflux'] / source['mw_transmission_g'])
    r_truth = 22.5 - 2.5 * np.log10(source['sim_rflux'] / source['mw_transmission_r'])
    z_truth = 22.5 - 2.5 * np.log10(source['sim_zflux'] / source['mw_transmission_z'])
    w1_truth = source['sim_w1']
    #w2_truth = source['sim_w2']
    g_measure = 22.5 - 2.5 * np.log10(source['flux_g'] / source['mw_transmission_g'])
    r_measure = 22.5 - 2.5 * np.log10(source['flux_r'] / source['mw_transmission_r'])
    z_measure = 22.5 - 2.5 * np.log10(source['flux_z'] / source['mw_transmission_z'])
    w1_measure = 22.5 - 2.5 * np.log10(source['flux_w1'] / source['mw_transmission_w1'])
    w2_measure = 22.5 - 2.5 * np.log10(source['flux_w2'] / source['mw_transmission_w2'])
    
    gflux_truth = source['sim_gflux']
    rflux_truth = source['sim_rflux']
    zflux_truth = source['sim_zflux']
    w1flux_truth = 10**((22.5 - source['sim_w1'])/2.5)*source['mw_transmission_w1']
    #w2flux_truth = 10**((22.5 - source['sim_w2'])/2.5)*source['mw_transmission_w2']
    
    gflux_measure = source['flux_g']
    rflux_measure = source['flux_r']
    zflux_measure = source['flux_z']
    w1flux_measure = source['flux_w1']
    #w2flux_measure = source['flux_w2']
    
    matplotlib.rc('xtick', labelsize=12)
    plt.rcParams.update({'font.size': 13})
    
    plt.figure(figsize=(9,18))  
    plt.subplot(5,2,1) 

    flux_plot_sigma((gflux_measure - gflux_truth)*np.sqrt(source['flux_ivar_g']),f_type = 'g')
    plt.subplot(5,2,3)    
    flux_plot_sigma((rflux_measure - rflux_truth)*np.sqrt(source['flux_ivar_r']),f_type = 'r')
    plt.subplot(5,2,5)    
    flux_plot_sigma((zflux_measure - zflux_truth)*np.sqrt(source['flux_ivar_z']),f_type = 'z')
    plt.subplot(5,2,7)
    flux_plot_sigma((w1flux_measure - w1flux_truth)*np.sqrt(source['flux_ivar_w1']),f_type = 'w1')
    #plt.subplot(5,2,8)
    #flux_plot_sigma((w2flux_measure - w2flux_truth)/np.sqrt(source['flux_ivar_w2']),f_type = 'w2')
    
    plt.subplot(5,2,2)
    one_sigma_2dplot((gflux_measure - gflux_truth)*np.sqrt(source['flux_ivar_g']), g_measure, title_x = 'g true',title_y = r'$\Delta$ g flux / err (output-input)')
    plt.subplot(5,2,4)
    one_sigma_2dplot((rflux_measure - rflux_truth)*np.sqrt(source['flux_ivar_r']), r_measure, title_x = 'r true',title_y = r'$\Delta$ r flux / err (output-input)')
    plt.subplot(5,2,6)
    one_sigma_2dplot((zflux_measure - zflux_truth)*np.sqrt(source['flux_ivar_z']), z_measure, title_x = 'z true',title_y = r'$\Delta$ z flux / err (output-input)')
    plt.subplot(5,2,8)
    one_sigma_2dplot((w1flux_measure - w1flux_truth)*np.sqrt(source['flux_ivar_w1']), w1_measure, title_x = 'w1 true',title_y = r'$\Delta$ w1 / err flux (output-input)')
    #plt.subplot(5,2,10)
    #one_sigma_2dplot(w2flux_measure - w2flux_truth, w2_measure, title_x = 'delta w2 flux',title_y = r'$\Delta$ w2 (output-input)')
    
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1b.png')

    
    

def fig1_cosmos(catalog,startid):
    
    source = catalog.processed_one
    sel = (source['matched_cosmos'])&(source['maskbits']==0)&(source['matched'])
    source = source[sel]
    #truth1: input truth
    g_truth1 = 22.5 - 2.5 * np.log10(source['sim_gflux'] / source['mw_transmission_g'])
    r_truth1 = 22.5 - 2.5 * np.log10(source['sim_rflux'] / source['mw_transmission_r'])
    z_truth1 = 22.5 - 2.5 * np.log10(source['sim_zflux'] / source['mw_transmission_z'])
    w1_truth1 = source['sim_w1']
    #w2_truth = source['sim_w2']
    
    #truth 2: mean of 10 validations
    None_flag = None
    for one_set in range(80,90):
        if None_flag is None:
            g_truth2 = 22.5 - 2.5 * np.log10(source['set_%d_flux_g'%one_set] / source['mw_transmission_g'])
            r_truth2 = 22.5 - 2.5 * np.log10(source['set_%d_flux_r'%one_set] / source['mw_transmission_r'])
            z_truth2 = 22.5 - 2.5 * np.log10(source['set_%d_flux_z'%one_set] / source['mw_transmission_z'])
            w1_truth2 = 22.5 - 2.5 * np.log10(source['set_%d_flux_w1'%one_set] / source['mw_transmission_w1'])
            #w2_truth2 = 22.5 - 2.5 * np.log10(source['set_%s_flux_w1'%one_set] / source['mw_transmission_w2'])
            None_flag = True
        else:
            g_truth2 += 22.5 - 2.5 * np.log10(source['set_%d_flux_g'%one_set] / source['mw_transmission_g'])
            r_truth2 += 22.5 - 2.5 * np.log10(source['set_%d_flux_r'%one_set] / source['mw_transmission_r'])
            z_truth2 += 22.5 - 2.5 * np.log10(source['set_%d_flux_z'%one_set] / source['mw_transmission_z'])
            w1_truth2 += 22.5 - 2.5 * np.log10(source['set_%d_flux_w1'%one_set] / source['mw_transmission_w1'])
            #w2_truth2 = 22.5 - 2.5 * np.log10(source['set_%s_flux_w1'%one_set] / source['mw_transmission_w2'])
    g_truth2 = g_truth2/10.
    r_truth2 = r_truth2/10.
    z_truth2 = z_truth2/10.
    w1_truth2 = w1_truth2/10.
    
    #difference between obiwan truth and g truth
    plt.figure(figsize=(9,18))  
    plt.subplot(5,2,1)    
    plt.title('input mag - mean output mag')

    one_sigma_2dplot(g_truth1 - g_truth2, g_truth1,title_x = 'g true',title_y = r'$\Delta$ g (input-output)')
    plt.subplot(5,2,3)    
    one_sigma_2dplot(r_truth1 - r_truth2, r_truth1,title_x = 'r true',title_y = r'$\Delta$ r (input-output)')
    plt.subplot(5,2,5)    
    one_sigma_2dplot(z_truth1 - z_truth2, z_truth1,title_x = 'z true',title_y = r'$\Delta$ z (input-output)')
    plt.subplot(5,2,7)
    one_sigma_2dplot(w1_truth1 - w1_truth2, w1_truth1,title_x = 'w1 true',title_y = r'$\Delta$ w1 (input-output)')
    #plt.subplot(5,2,8)
    #one_sigma_2dplot(w2_truth-w2_measure,w2_measure,title_x = 'w2 true',title_y = r'$\Delta$ w2 (input-output)')
    
    
    g_measure = 22.5 - 2.5 * np.log10(source['flux_g'] / source['mw_transmission_g'])
    r_measure = 22.5 - 2.5 * np.log10(source['flux_r'] / source['mw_transmission_r'])
    z_measure = 22.5 - 2.5 * np.log10(source['flux_z'] / source['mw_transmission_z'])
    w1_measure = 22.5 - 2.5 * np.log10(source['flux_w1'] / source['mw_transmission_w1'])
    #w2_measure = 22.5 - 2.5 * np.log10(source['flux_w2'] / source['mw_transmission_w2'])
    
    plt.subplot(5,2,2)
    plt.title('input mag - output mag')
    one_sigma_2dplot(g_truth1 - g_measure, g_truth1, title_x = 'g true',title_y = r'$\Delta$ g (input-output)')
    plt.subplot(5,2,4)
    one_sigma_2dplot(r_truth1 - r_measure, r_truth1, title_x = 'r true',title_y = r'$\Delta$ r (input-output)')
    plt.subplot(5,2,6)
    one_sigma_2dplot(z_truth1 - z_measure, z_truth1, title_x = 'z true',title_y = r'$\Delta$ z (input-output)')
    plt.subplot(5,2,8)
    one_sigma_2dplot(w1_truth1 - w1_measure, w1_truth1, title_x = 'w1 true',title_y = r'$\Delta$ w1 (input-output)')
    #plt.subplot(5,2,10)
    #one_sigma_2dplot(w2flux_measure - w2flux_truth, w2_measure, title_x = 'delta w2 flux',title_y = r'$\Delta$ w2 (output-input)')
    
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1_cosmos.png')    
    

def fig1b_cosmos(catalog,band,set_num, obj = None,startid=None):
    # SECTION SIM
    source = catalog.processed_one
    #selecting tracer that comes in *any* sets as our 'potential' tracers
    if obj is not None:
        tracer = False
        for i in range(80,90):
            tracer|=source['set_%d_%s'%(i,obj)].copy()
    else:
            tracer = True
            
    cross_matched = True
    for i in range(80,90):
        cross_matched&=source['set_%s_matched'%i].copy()
    sel2 = (source['matched_cosmos'])&(source['matched'])&(source['maskbits']==0)&tracer&cross_matched
    source2 = source[sel2]
    print("%d tracers in set %s"%(len(source2), set_num))
    #true flux from sim
    if band!='w1':
        flux_sim = source2['sim_%sflux'%band].copy()
    else:
        flux_sim = 10**((22.5 - source2['sim_w1'])/2.5)*source2['mw_transmission_w1'].copy()
    flux_measure2 = source2['flux_%s'%band].copy()

    None_flag = None
    for i in range(80,90):
        if None_flag is None:
            flux_ave2 = source2['set_%d_flux_%s'%(i, band)].copy()
            None_flag = True
        else:
            flux_ave2 += source2['set_%d_flux_%s'%(i, band)].copy()
    flux_ave2 = flux_ave2/10.
    
    matplotlib.rc('xtick', labelsize=12)
    plt.rcParams.update({'font.size': 13})
    
    plt.figure(figsize=(15,10))  
    plt.subplot(3,2,1) 
    one_subplot_hist(variable=np.array(flux_sim), xlabel = 'flux %s sim'%band, ylabel = 'PDF', bins=30)
    
    plt.subplot(3,2,2)
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
    n = int(int(catalog.subsection[-2:])-80)
    print(n)
    dat = fits.getdata(topdir+'cosmos_set%d.fits'%n)
    cross_match = True
    for i in range(10):
        try:
            tmp = dat['set_%d_matched'%i].copy()
        except:
            continue
        cross_match &= dat['set_%d_matched'%i].copy()
    cosmos_LRG = False
    for i in range(10):
        cosmos_LRG |= dat['set_%d_%s'%(i,obj)].copy()&dat['set_%d_isfaint'%i].copy()
    sel_cosmos = cross_match&cosmos_LRG
    dat1 = dat[sel_cosmos]

    flux_measure1 = dat1['flux_%s'%band].copy()
    None_flag = None
    for i in range(10):
        if None_flag is None:
            try:
                flux_ave1 = dat1['set_%d_flux_%s'%(i,band)].copy()
                None_flag = True
            except:
                flux_ave1 = dat1['flux_%s'%(band)].copy()
                None_flag = True
            
        else:
            try:
                flux_ave1 += dat1['set_%d_flux_%s'%(i,band)].copy()
            except:
                flux_ave1 += dat1['flux_%s'%(band)].copy()
    flux_ave1 = flux_ave1/10.
    plt.title('real this set & ave')

    one_subplot_hist(variable=np.array(flux_measure1), xlabel = 'this set %s flux'%band, ylabel = 'PDF', bins=30)
    one_subplot_hist(variable=np.array(flux_ave1), xlabel = 'ave %s flux'%band, ylabel = 'PDF', bins=30, color = 'r')
    
    
    plt.subplot(3,2,3)
    plt.title('sim_out this set & ave')
    one_subplot_hist(variable=np.array(flux_measure2), xlabel = 'this set %s flux'%band, ylabel = 'PDF', bins=30)
    one_subplot_hist(variable=np.array(flux_ave2), xlabel = 'ave %s flux'%band, ylabel = 'PDF', bins=30, color = 'r')
   
    
    plt.subplot(3,2,4)
    plt.title('real this set - ave')
    flux_plot_sigma((flux_measure1 - flux_ave1)*np.sqrt(dat1['flux_ivar_%s'%band]),f_type = band)
    
    plt.subplot(3,2,5)
    plt.title('sim this set - truth')
    flux_plot_sigma((flux_measure2 - flux_sim)*np.sqrt(source2['flux_ivar_%s'%band]),f_type = band)
    
    plt.subplot(3,2,6)
    plt.title('sim this set - ave')
    flux_plot_sigma((flux_measure2 - flux_ave2)*np.sqrt(source2['flux_ivar_%s'%band]),f_type = band)
    
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1b_cosmos_%s.png'%band)    
    
    
#next mophorlogy plots 
#e1,e2,shape_r, sersic n
def fig1c(catalog,startid):
    source = catalog.processed_one
    sel = source['matched']&(source['maskbits']==0)
    source = source[sel]
    e1_truth = source['sim_e1']
    e2_truth = source['sim_e2']
    r_truth = source['sim_rhalf']
    sersic_truth = source['sim_sersic_n']
    e_truth = np.sqrt(e1_truth**2+e2_truth**2)
    e1_measure = source['shape_e1']
    e2_measure = source['shape_e2']
    r_measure  = source['shape_r']
    e_measure = np.sqrt(e1_measure**2+e2_measure**2)
    sersic_measure = source['sersic']

    matplotlib.rc('xtick', labelsize=12)
    plt.rcParams.update({'font.size': 13})
    
    plt.figure(figsize=(9,18))
    plt.subplot(4,2,1) 
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e_measure), xlabel = 'e', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e_truth), xlabel = 'e', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e truth',color = 'r',percentile=0)
    plt.subplot(4,2,3)    
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_measure), xlabel = 'r', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'r measure',percentile=1)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_truth), xlabel = 'r', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'r truth',color = 'r',percentile=1)
    plt.subplot(4,2,5)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_measure), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'sersic measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'sersic truth', color='r',percentile=0)
    plt.subplot(4,2,7)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_measure - e1_truth), xlabel = 'e1,e2', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'delta e1',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e2_measure - e2_truth), xlabel = 'e1,e2', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'delta e2',color = 'r',percentile=0)
    
    plt.subplot(4,2,2)
    sel = (e_truth>0)
    one_sigma_2dplot(e_measure[sel] - e_truth[sel], e_truth[sel], title_x = 'Eclipticity (e_true>0)',title_y = 'e measure - truth')
    plt.subplot(4,2,4)
    sel = (r_truth>0)
    one_sigma_2dplot(((r_measure[sel] - r_truth[sel])), r_measure[sel], title_x = 'shape r (r_true>0)',title_y = 'shape r (measure - truth)')
    plt.subplot(4,2,6)
    sel = e1_truth>0
    one_sigma_2dplot(((e1_measure[sel] - e1_truth[sel])), e1_truth[sel], title_x = 'e1 true（>0 only)',title_y = 'e1 (measure - truth)')
    plt.subplot(4,2,8)
    sel = e2_truth>0
    one_sigma_2dplot(((e2_measure[sel] - e2_truth[sel])), e2_truth[sel], title_x = 'e2 true（>0 only)',title_y = 'e2 (measure - truth)')
    
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1c.png')

    
    
    
def fig1d(catalog,startid):
    source = catalog.processed_one
    
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/'
    dat = fits.getdata(topdir+'/preseed.fits')
    sel1 = dat['LRG_sv3_like_cross']&(dat['truth_galdepth_z']>1000)
    flux_g = dat['truth_flux_g']/dat['truth_mw_transmission_g']
    flux_r = dat['truth_flux_r']/dat['truth_mw_transmission_r']
    flux_z = dat['truth_flux_z']/dat['truth_mw_transmission_z']
    flux_w1 = dat['flux_w1']/dat['mw_transmission_w1']
    flux_w2 = dat['flux_w2']/dat['mw_transmission_w2']

    flux_g = flux_g
    flux_r = flux_r
    flux_z = flux_z
    flux_w1 = flux_w1

    gmag = 22.5 - 2.5 * np.log10(flux_g)
    rmag = 22.5 - 2.5 * np.log10(flux_r)
    zmag = 22.5 - 2.5 * np.log10(flux_z)
    w1mag = 22.5 - 2.5 * np.log10(flux_w1)
    
    sel2 = (gmag>18)&(rmag>18)&(zmag>18)&(w1mag>18)&(dat['truth_shape_r']<6)
    
    dr9_dat = dat[sel1&sel2]
    
    dr9_e1 = dr9_dat['shape_e1']
    dr9_e2 = dr9_dat['shape_e2']
    dr9_r = dr9_dat['shape_r']
    dr9_sersic = dr9_dat['sersic']
    dr9_e = np.sqrt(dr9_e1**2+dr9_e2**2)
    
    dr9_e1_truth = dr9_dat['truth_shape_e1']
    dr9_e2_truth = dr9_dat['truth_shape_e2']
    dr9_r_truth = dr9_dat['truth_shape_r']
    dr9_sersic_truth = dr9_dat['truth_sersic']
    dr9_e_truth = np.sqrt(dr9_e1_truth**2+dr9_e2_truth**2)
    
    sel = source['matched']&(source['maskbits']==0)
    source = source[sel]
    e1_truth = source['sim_e1']
    e2_truth = source['sim_e2']
    r_truth = source['sim_rhalf']
    sersic_truth = source['sim_sersic_n']
    e_truth = np.sqrt(e1_truth**2+e2_truth**2)
    e1_measure = source['shape_e1']
    e2_measure = source['shape_e2']
    r_measure  = source['shape_r']
    e_measure = np.sqrt(e1_measure**2+e2_measure**2)
    sersic_measure = source['sersic']
    
    
    matplotlib.rc('xtick', labelsize=12)
    plt.rcParams.update({'font.size': 13})
    
    plt.figure(figsize=(9,18))
    plt.subplot(4,2,1) 
    
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e_truth), xlabel = 'e', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e_measure), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e dr9',color = 'y',percentile=0)
    plt.subplot(4,2,3)    
    
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_truth), xlabel = 'r', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'r truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_measure), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_r), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r dr9',color = 'y',percentile=0)
    
    plt.subplot(4,2,5)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'sersic truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_measure), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_sersic), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic dr9', color='y',percentile=0)
    plt.subplot(4,2,7)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e1 truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_measure), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 dr9 measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e1), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 dr9', color='y',percentile=0)
    
    plt.subplot(4,2,2)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e_truth), xlabel = 'e', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e_truth), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e dr9',color = 'y',percentile=0)
    
    plt.subplot(4,2,4)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_r_truth), xlabel = 'r', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'r seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_truth), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_r), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r dr9',color = 'y',percentile=0)
    
    plt.subplot(4,2,6)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'sersic seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_sersic), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic dr9', color='y',percentile=0)
    plt.subplot(4,2,8)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e1 seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e1), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 dr9', color='y',percentile=0)
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1d.png')

    
    
    
def fig1e_cosmos(catalog,set_num, startid):
    #instead of dr9, use sources from cosmos repeats
    source = catalog.processed_one
    
    #get dr9 sources 
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/'
    dat = fits.getdata(topdir+'/preseed.fits')
    sel1 = dat['LRG_sv3_like_cross']&(dat['truth_galdepth_z']>1000)
    flux_g = dat['truth_flux_g']/dat['truth_mw_transmission_g']
    flux_r = dat['truth_flux_r']/dat['truth_mw_transmission_r']
    flux_z = dat['truth_flux_z']/dat['truth_mw_transmission_z']
    flux_w1 = dat['flux_w1']/dat['mw_transmission_w1']
    flux_w2 = dat['flux_w2']/dat['mw_transmission_w2']

    flux_g = flux_g
    flux_r = flux_r
    flux_z = flux_z
    flux_w1 = flux_w1

    gmag = 22.5 - 2.5 * np.log10(flux_g)
    rmag = 22.5 - 2.5 * np.log10(flux_r)
    zmag = 22.5 - 2.5 * np.log10(flux_z)
    w1mag = 22.5 - 2.5 * np.log10(flux_w1)
    
    sel2 = (gmag>18)&(rmag>18)&(zmag>18)&(w1mag>18)&(dat['truth_shape_r']<6)
    
    dr9_dat = dat[sel1&sel2]
    
    dr9_e1 = dr9_dat['shape_e1']
    dr9_e2 = dr9_dat['shape_e2']
    dr9_r = dr9_dat['shape_r']
    dr9_sersic = dr9_dat['sersic']
    dr9_e = np.sqrt(dr9_e1**2+dr9_e2**2)
    
    dr9_e1_truth = dr9_dat['truth_shape_e1']
    dr9_e2_truth = dr9_dat['truth_shape_e2']
    dr9_r_truth = dr9_dat['truth_shape_r']
    dr9_sersic_truth = dr9_dat['truth_sersic']
    dr9_e_truth = np.sqrt(dr9_e1_truth**2+dr9_e2_truth**2)
    
    #cosmos repeats lrg_sv3_like sources
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
    dat = fits.getdata(topdir+'/cosmos_set%d.fits'%(int(set_num)-80))
    sel = dat['lrg_sv3_like']
    flux_g = dat['flux_g']/dat['mw_transmission_g']
    flux_r = dat['flux_r']/dat['mw_transmission_r']
    flux_z = dat['flux_z']/dat['mw_transmission_z']
    flux_w1 = dat['flux_w1']/dat['mw_transmission_w1']

    flux_g = flux_g
    flux_r = flux_r
    flux_z = flux_z
    flux_w1 = flux_w1

    gmag = 22.5 - 2.5 * np.log10(flux_g)
    rmag = 22.5 - 2.5 * np.log10(flux_r)
    zmag = 22.5 - 2.5 * np.log10(flux_z)
    w1mag = 22.5 - 2.5 * np.log10(flux_w1)
    
    sel2 = (gmag>18)&(rmag>18)&(zmag>18)&(w1mag>18)&(dat['shape_r']<6)
    cosmos_dat = dat[sel&sel2]
    cosmos_e1 = cosmos_dat['shape_e1']
    cosmos_e2 = cosmos_dat['shape_e1']
    cosmos_r = cosmos_dat['shape_r']
    cosmos_e = np.sqrt(cosmos_e1**2+cosmos_e2**2)
    cosmos_sersic = cosmos_dat['sersic']
    
    
    
    sel = source['matched']&(source['maskbits']==0)
    source = source[sel]
    e1_truth = source['sim_e1']
    e2_truth = source['sim_e2']
    r_truth = source['sim_rhalf']
    sersic_truth = source['sim_sersic_n']
    e_truth = np.sqrt(e1_truth**2+e2_truth**2)
    e1_measure = source['shape_e1']
    e2_measure = source['shape_e2']
    r_measure  = source['shape_r']
    e_measure = np.sqrt(e1_measure**2+e2_measure**2)
    sersic_measure = source['sersic']
    
    
    matplotlib.rc('xtick', labelsize=12)
    plt.rcParams.update({'font.size': 13})
    
    plt.figure(figsize=(9,18))
    
    N_bins = 30
    plt.subplot(4,2,1) 
    
    delta_flux, max_bin, bins_n_1, bins = one_subplot_hist(variable=np.array(e_truth), xlabel = 'e', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(e_measure), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e measure',percentile=0)
    delta_flux, max_bin, bins_n_3, bins = one_subplot_hist(variable=np.array(cosmos_e), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e cosmos',color = 'y',percentile=0)
    
    #ks test measure vs cosmos
    ks = (bins_n_2-bins_n_3)
    
    
    plt.subplot(4,2,3)    
    
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_truth), xlabel = 'r', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'r truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_measure), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(cosmos_r), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r cosmos',color = 'y',percentile=0)
    
    plt.subplot(4,2,5)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'sersic truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_measure), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(cosmos_sersic), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic cosmos', color='y',percentile=0)
    plt.subplot(4,2,7)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_truth), xlabel = 'e1', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e1 truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_measure), xlabel = 'e1', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 measure',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(cosmos_e1), xlabel = 'e1', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 cosmos', color='y',percentile=0)
    
    plt.subplot(4,2,2)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e_truth), xlabel = 'e', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e_truth), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e dr9',color = 'y',percentile=0)
    
    plt.subplot(4,2,4)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_r_truth), xlabel = 'r', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'r seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(r_truth), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r truth',color = 'r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_r), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r dr9',color = 'y',percentile=0)
    
    plt.subplot(4,2,6)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'sersic seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_sersic), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic dr9', color='y',percentile=0)
    plt.subplot(4,2,8)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=30, density = True, mean= False, label = 'e1 seed',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 truth', color='r',percentile=0)
    delta_flux, max_bin, bins_n, bins = one_subplot_hist(variable=np.array(dr9_e1), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 dr9', color='y',percentile=0)
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1e_cosmos.png')
    
    
    
def fig1e_cosmos_ks(catalog,set_num, startid):
    #KS test following Two-sample Kolmogorov–Smirnov test in https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    
    #instead of dr9, use sources from cosmos repeats
    source = catalog.processed_one
    
    #get dr9 sources 
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/'
    dat = fits.getdata(topdir+'/preseed.fits')
    #dat = fits.getdata("/global/homes/h/huikong/obiwan_analysis/py/lrg_truth.fits")
    sel1 = dat['LRG_sv3_like_cross']&(dat['truth_galdepth_z']>1000)
    flux_g = dat['truth_flux_g']/dat['truth_mw_transmission_g']
    flux_r = dat['truth_flux_r']/dat['truth_mw_transmission_r']
    flux_z = dat['truth_flux_z']/dat['truth_mw_transmission_z']
    flux_w1 = dat['flux_w1']/dat['mw_transmission_w1']
    flux_w2 = dat['flux_w2']/dat['mw_transmission_w2']

    flux_g = flux_g
    flux_r = flux_r
    flux_z = flux_z
    flux_w1 = flux_w1

    gmag = 22.5 - 2.5 * np.log10(flux_g)
    rmag = 22.5 - 2.5 * np.log10(flux_r)
    zmag = 22.5 - 2.5 * np.log10(flux_z)
    w1mag = 22.5 - 2.5 * np.log10(flux_w1)
    
    sel2 = (gmag>18)&(rmag>18)&(zmag>18)&(w1mag>18)&(dat['truth_shape_r']<6)
    
    dr9_dat = dat[sel1&sel2]
    
    dr9_e1 = dr9_dat['shape_e1']
    dr9_e2 = dr9_dat['shape_e2']
    dr9_r = dr9_dat['shape_r']
    dr9_sersic = dr9_dat['sersic']
    dr9_e = np.sqrt(dr9_e1**2+dr9_e2**2)
    
    dr9_e1_truth = dr9_dat['truth_shape_e1']
    dr9_e2_truth = dr9_dat['truth_shape_e2']
    dr9_r_truth = dr9_dat['truth_shape_r']
    dr9_sersic_truth = dr9_dat['truth_sersic']
    dr9_e_truth = np.sqrt(dr9_e1_truth**2+dr9_e2_truth**2)
    
    #cosmos repeats lrg_sv3_like sources
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
    dat = fits.getdata(topdir+'/cosmos_set%d.fits'%(int(set_num)-80))
    sel = dat['lrg_sv3_like']
    flux_g = dat['flux_g']/dat['mw_transmission_g']
    flux_r = dat['flux_r']/dat['mw_transmission_r']
    flux_z = dat['flux_z']/dat['mw_transmission_z']
    flux_w1 = dat['flux_w1']/dat['mw_transmission_w1']

    flux_g = flux_g
    flux_r = flux_r
    flux_z = flux_z
    flux_w1 = flux_w1

    gmag = 22.5 - 2.5 * np.log10(flux_g)
    rmag = 22.5 - 2.5 * np.log10(flux_r)
    zmag = 22.5 - 2.5 * np.log10(flux_z)
    w1mag = 22.5 - 2.5 * np.log10(flux_w1)
    
    sel2 = (gmag>18)&(rmag>18)&(zmag>18)&(w1mag>18)&(dat['shape_r']<6)
    cosmos_dat = dat[sel&sel2]
    cosmos_e1 = cosmos_dat['shape_e1']
    cosmos_e2 = cosmos_dat['shape_e2']
    cosmos_r = cosmos_dat['shape_r']
    cosmos_e = np.sqrt(cosmos_e1**2+cosmos_e2**2)
    cosmos_sersic = cosmos_dat['sersic']
    
    
    
    sel = source['matched']&(source['maskbits']==0)
    source = source[sel]
    e1_truth = source['sim_e1']
    e2_truth = source['sim_e2']
    r_truth = source['sim_rhalf']
    sersic_truth = source['sim_sersic_n']
    e_truth = np.sqrt(e1_truth**2+e2_truth**2)
    e1_measure = source['shape_e1']
    e2_measure = source['shape_e2']
    r_measure  = source['shape_r']
    e_measure = np.sqrt(e1_measure**2+e2_measure**2)
    sersic_measure = source['sersic']
    
    
    matplotlib.rc('xtick', labelsize=12)
    plt.rcParams.update({'font.size': 13})
    
    plt.figure(figsize=(9,18))
    
    N_bins = 30
    plt.subplot(4,2,1) 
    
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(e_measure), xlabel = 'e', ylabel = 'CDF', bins=N_bins, density = True, mean= False, label = 'e measure',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_3, bins = one_subplot_hist(variable=np.array(cosmos_e), xlabel = 'e', ylabel = 'CDF', bins=bins, density = True, mean= False, label = 'e cosmos',color = 'y',percentile=0,cdf=True)
    
    #ks test measure vs cosmos
    D_nm = np.max(np.abs(bins_n_2-bins_n_3))
    n = len(e_measure)
    m = len(cosmos_e)
    c_aphla = D_nm*np.sqrt((n*m/(n+m)))
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    plt.subplot(4,2,3)    
    
    
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(r_measure), xlabel = 'r', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'r measure',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_3, bins = one_subplot_hist(variable=np.array(cosmos_r), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r cosmos',color = 'y',percentile=0,cdf=True)
    
    D_nm = np.max(np.abs(bins_n_2-bins_n_3))
    n = len(e_measure)
    m = len(cosmos_e)
    c_aphla = D_nm*np.sqrt((n*m/(n+m)))
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    plt.subplot(4,2,5)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(sersic_measure), xlabel = 'sersic', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'sersic measure',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_3, bins = one_subplot_hist(variable=np.array(cosmos_sersic), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic cosmos', color='y',percentile=0,cdf=True)
    
    D_nm = np.max(np.abs(bins_n_2-bins_n_3))
    n = len(sersic_measure)
    m = len(cosmos_sersic)
    c_aphla = D_nm*np.sqrt((n*m/(n+m)))
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    plt.subplot(4,2,7)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(e1_measure), xlabel = 'e1', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'e1 measure',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_3, bins = one_subplot_hist(variable=np.array(cosmos_e1), xlabel = 'e1', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 cosmos', color='y',percentile=0,cdf=True)
    
    D_nm = np.max(np.abs(bins_n_2-bins_n_3))
    n = len(e1_measure)
    m = len(cosmos_e1)
    c_aphla = D_nm*np.sqrt((n*m/(n+m)))
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    plt.subplot(4,2,2)
    delta_flux, max_bin, bins_n_1, bins = one_subplot_hist(variable=np.array(dr9_e_truth), xlabel = 'e', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'e seed',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(e_truth), xlabel = 'e', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e truth',color = 'r',percentile=0,cdf=True)
    
    D_nm = np.max(np.abs(bins_n_1-bins_n_2))
    n = len(dr9_e_truth)
    m = len(e_truth)
    c_aphla = D_nm*np.sqrt(m)
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    
    
    plt.subplot(4,2,4)
    delta_flux, max_bin, bins_n_1, bins = one_subplot_hist(variable=np.array(dr9_r_truth), xlabel = 'r', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'r seed',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(r_truth), xlabel = 'r', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'r truth',color = 'r',percentile=0,cdf=True)
    D_nm = np.max(np.abs(bins_n_1-bins_n_2))
    n = len(dr9_r_truth)
    m = len(r_truth)
    c_aphla = D_nm*np.sqrt(m)
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    
    plt.subplot(4,2,6)
    delta_flux, max_bin, bins_n_1, bins = one_subplot_hist(variable=np.array(dr9_sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'sersic seed',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(sersic_truth), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'sersic truth', color='r',percentile=0,cdf=True)
    
    D_nm = np.max(np.abs(bins_n_1-bins_n_2))
    n = len(dr9_sersic_truth)
    m = len(sersic_truth)
    c_aphla = D_nm*np.sqrt(m)
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    plt.subplot(4,2,8)
    delta_flux, max_bin, bins_n_1, bins = one_subplot_hist(variable=np.array(dr9_e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=N_bins, density = True, mean= False, label = 'e1 seed',percentile=0,cdf=True)
    delta_flux, max_bin, bins_n_2, bins = one_subplot_hist(variable=np.array(e1_truth), xlabel = 'sersic', ylabel = 'PDF', bins=bins, density = True, mean= False, label = 'e1 truth', color='r',percentile=0,cdf=True)
    
    D_nm = np.max(np.abs(bins_n_1-bins_n_2))
    n = len(dr9_e1_truth)
    m = len(e1_truth)
    c_aphla = D_nm*np.sqrt(m)
    alpha = 2*np.exp(-2*(c_aphla**2))
    plt.title("ks alpha=%.3f"%alpha)
    
    plt.tight_layout()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    if os.path.isdir(topdir) is False:
        subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig1e_cosmos_ks.png')