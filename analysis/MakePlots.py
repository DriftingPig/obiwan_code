import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np


class BasePlot(object):
	def __init__(self, **kwargs):
		self.BasePlotFlag = True
		super().__init__(**kwargs)
	def loadimg(self, img_fn):
		self.img = fits.getdata(img_fn)
	def imshow(self,img=None, vmax=None,vmin=None,percentile=None):
			if vmax is not None:
				assert(vmin is not None)
				plt.imshow(img,vmax=vmax,vmin=vmin)
			else:
				plt.imshow(img)

	@staticmethod
	def one_sigma_2dplot(x_value,y_value, title_x=None, title_y=None, x_boundary_percentile=1, percentile_y=16, y_boundary=5,bins=40):
		'''
		2d density plot with 3 percentile line drew on the plot
		'''
		sel = (x_value>=-1e9)&(x_value<=10e9)
		x_value = x_value[sel]
		y_value = y_value[sel]
		lower = np.percentile(x_value,x_boundary_percentile)
		higher = np.percentile(x_value,100-x_boundary_percentile)
		sel2 = (x_value>lower)&(x_value<higher)
		variable = variable[sel2]
		flux_diff = flux_diff[sel2]
        
		y = y_value
		x = x_value
    
		plt.xlabel(title)
		plt.ylabel('delta flux/err')
		yl,xl = _get_percentile(y,x,bins,100-percentile_y)
		plt.plot(xl,yl,color = 'y')
		l,xl = _get_percentile(y,x,bins,50)
		plt.plot(xl,yl,color = 'y')
		yl,xl = _get_percentile(y,x,bins,percentile_y)
		lt.plot(xl,yl,color = 'y')
		
		if y_boundary is not None:
			sel3 = (y_value>-y_boundary)&(y_value<y_boundary)
			y2 = y_value[sel3]
			x2 = x_value[sel3]
		else:
			y2 = y_value
			x2 = x_value

		h,xe,ye,i = plt.hist2d(x2,y2,bins=40,cmap = 'Blues')
		plt.plot(x,[0]*len(x),'r:',alpha=0.5)

		plt.set_xlim((x2.min(),x2.max()))
		plt.set_ylim((y2.min(),y2.max()))

	@staticmethod
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

	@staticmethod
	def one_subplot_hist(title,variable,xlabel = None,ylabel=None,percentile=5,bins=50,density=False):
		sel = (variable>=-1e-9)&(variable<=10e9)
		variable = variable[sel]
		ower = np.percentile(variable,percentile)
		higher = np.percentile(variable,100-percentile)
		sel2 = (variable>lower)&(variable<higher)
		variable = variable[sel2]
		n,bins,_ = plt.hist(variable,bins=bins,density=density)
		y1_cen = int(n.max()*3/4+0.5)
		y2_cen = int(n.max()*5/8+0.5)
		x_cen = bins[int(len(bins)*5/8+0.5)]
		plt.text(x_cen,y1_cen,"mean:%.3f"%variable.mean())
		plt.text(x_cen,y2_cen,"std:%.3f"%variable.std())
		if xlabel is not None:
			plt.xlabel(xlabel)
		if ylabel is not None:
			plt.ylabel(ylabel)
		if title is not None:
			plt.title(title)

