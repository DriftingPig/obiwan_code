class cuts_record(object):
    def __init__(self,cut_type):
        exec("self.%s()"%cut_type)
        self.cut_type = cut_type
        self.params()
        

    def _work_string(self):
        #add these lines to each color cut you make
        L = locals()
        L.pop("self")
        rtn = dict([(k,L[k]) for k in L.keys()])
        keys = []
        for key in rtn.keys():
            keys.append(key)
            setattr(self,key,eval(key))
    def params(self):
        step = 0.01
        change = 0.004
        lower_limit = 0.025
        savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cuts_info_%s_deep.fits'%self.cut_type
        L = locals()
        L.pop("self")
        rtn = dict([(k,L[k]) for k in L.keys()])
        for key in rtn.keys():
            self.keys.append(key)
            setattr(self,key,eval(key))
            
    def LRG_sv3(self):
        cut1_origin = 'zmag - w1mag > 0.8 * (rmag - zmag) - 0.6'
        cut2_origin = 'zfibermag < 21.7'
        #cut3 = cut3_1&cut3_2|cut3_3
        cut3_1_origin = 'gmag - rmag > 1.3'
        cut3_2_origin = '(gmag - rmag) > -1.55 * (rmag - w1mag)+3.13'
        cut3_3_origin = 'rmag - w1mag > 1.8'
        cut4_1_origin = 'rmag - w1mag > (w1mag - 17.26) * 1.8'
        cut4_2_origin = 'rmag - w1mag > (w1mag - 16.36) * 1.'
        cut4_3_origin = 'rmag - w1mag > 3.29'
        equ_cut_all = "'(%s)&(%s)&((%s)&(%s)|(%s))&((%s)&(%s)|(%s))'%(cut1_origin, cut2_origin, cut3_1_origin, cut3_2_origin,cut3_3_origin,cut4_1_origin,cut4_2_origin,cut4_3_origin)"
        cut_all = eval(equ_cut_all)
        cuts = ['cut1_origin','cut2_origin','cut3_1_origin','cut3_2_origin','cut3_3_origin','cut4_1_origin','cut4_2_origin','cut4_3_origin']
        symbols = ['-',   '+',   '-',     '-',     '-',      '-',     '-',     '-']
        
        L = locals()
        L.pop("self")
        rtn = dict([(k,L[k]) for k in L.keys()])
        keys = []
        for key in rtn.keys():
            keys.append(key)
            setattr(self,key,eval(key))
        self.keys = keys
        
        #self.target_type = 'sv3_lrg'
        self.target_type = 'lrg_sv3'
        
    def LRG_baseline(self):
        cut1_origin = 'zmag - w1mag > 0.8 * (rmag-zmag) - 0.6'
        cut2_origin = 'zfibermag < 21.5'
        cut3_origin = 'rmag - w1mag > 1.1'
        cut4_origin = 'rmag - w1mag > (w1mag - 17.22) * 1.8 '
        cut5_origin = 'rmag - w1mag > w1mag - 16.37'
        equ_cut_all = "'(%s)&(%s)&(%s)&(%s)&(%s)'%(cut1_origin, cut2_origin, cut3_origin, cut4_origin, cut5_origin)"
        cut_all = eval(equ_cut_all)
        cuts = ['cut1_origin','cut2_origin','cut3_origin','cut4_origin','cut5_origin']
        symbols = ['-',   '+',   '-',     '-',     '-']
        
        L = locals()
        L.pop("self")
        rtn = dict([(k,L[k]) for k in L.keys()])
        keys = []
        for key in rtn.keys():
            keys.append(key)
            setattr(self,key,eval(key))
        self.keys = keys
        
        self.target_type = 'lrgir'

    def ELG_sv3(self):
        cut1_origin = 'gfibermag<24.1'
        cut2_origin = 'gmag - rmag < 0.5*(rmag - zmag) + 0.1'
        cut3_origin = 'gmag - rmag < -1.2*(rmag - zmag) + 1.6'
        cut4_origin = 'gmag - rmag >= -1.2*(rmag - zmag) + 1.3'
        equ_cut_all = "'(%s)&(%s)&(%s)&(%s)'%(cut1_origin, cut2_origin, cut3_origin, cut4_origin)"
        cut_all = eval(equ_cut_all)
        cuts =  ['cut1_origin','cut2_origin','cut3_origin','cut4_origin']
        symbols = ['+',   '+',   '+',     '-']
        
        L = locals()
        L.pop("self")
        rtn = dict([(k,L[k]) for k in L.keys()])
        keys = []
        for key in rtn.keys():
            keys.append(key)
            setattr(self,key,eval(key))
        self.keys = keys
        
        self.target_type = 'sv3_elg'
    def ELG_hip_sv3(self):
        cut1_origin = 'gfibermag<24.1'
        cut2_origin = 'gmag - rmag < 0.5*(rmag - zmag) + 0.1'
        cut3_origin = 'gmag - rmag < -1.2*(rmag - zmag) + 1.3'
        equ_cut_all = "'(%s)&(%s)&(%s)'%(cut1_origin, cut2_origin, cut3_origin)"
        cut_all = eval(equ_cut_all)
        cuts =  ['cut1_origin','cut2_origin','cut3_origin']
        symbols = ['+',   '+',   '+']

        L = locals()
        L.pop("self")
        rtn = dict([(k,L[k]) for k in L.keys()])
        keys = []
        for key in rtn.keys():
            keys.append(key)
            setattr(self,key,eval(key))
        self.keys = keys

        self.target_type = 'sv3_elg_hip'

