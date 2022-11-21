import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from kapteyn import kmpfit
import pandas
import matplotlib.pyplot as plt
import warnings
plt.rcParams['figure.dpi'] = 200



class Timelapse:
    def __init__(self,isolate_name):
        self.isolate_name = isolate_name

        self.concentration_maps = {}
        self.concentration_averages = {}
        self.concentration_average_stderr = {}

        self.AUC_averages = {}
        self.AUC_average_stderr = {}

        self.timepoints = None
        self.blanks = None

    def add_concentration(self,concentration):
        if concentration not in self.concentration_maps:
            self.concentration_maps[concentration] = {}

    def add_tseries(self,tseries, repeat_ID = None, concentration = None):
        assert repeat_ID is not None and concentration is not None

        if concentration not in self.concentration_maps:
            raise ValueError('Concentration step not detected')

        if repeat_ID in self.concentration_maps[concentration]:
            raise ValueError('Repeat ID already exists in concentration {}'.format(concentration))

        self.concentration_maps[concentration][repeat_ID] = tseries

    def add_timepoints(self,timepoints):
        '''Add x axis, the timepoints'''
        self.timepoints = timepoints

    def add_blanks(self,tseries):
        print('{} : Detected {} blanks, averaging.'.format(self.isolate_name,len(tseries)))
        self.blanks = np.asarray(tseries)
        self.blanks = np.mean(self.blanks,axis=0)


    def calculate_averages(self):

        assert self.blanks is not None

        for concentration,repeats in self.concentration_maps.items():

            count = len(repeats)
            print('{} : Averaging {} repeats of {}'.format(self.isolate_name,count,concentration))

            reps = []
            for repID, rep in repeats.items():
                reps.append(rep)

            reps = np.asarray(reps)

            #subtract blank
            reps = reps - self.blanks

            av = np.mean(reps,axis=0)
            stdev = np.std(reps,axis=0)


            #store
            self.concentration_averages[concentration] = av
            self.concentration_average_stderr[concentration] = stdev/np.sqrt(count)

    def plot_averages(self,sampling_interval=None,ax=None,**kwargs):
        '''If sampling_interval is None, autopick one based on timepoint density '''
        assert self.concentration_averages is not {} and self.concentration_average_stderr is not {}

        if sampling_interval is None:
            interval = int(len(self.timepoints)/20) #Downsample to total of 20 points
        else:
            interval = sampling_interval

        if ax is None:
            fig,ax = plt.subplots(1,1)
        else:
            ax=ax

        for conc,tlapse in self.concentration_averages.items():
            err = self.concentration_average_stderr[conc]

            ax.errorbar(self.timepoints[0::interval], tlapse[0::interval], yerr=err[0::interval], capsize=2, label=str(conc),**kwargs)
            ax.legend()
        ax.set_title(self.isolate_name)


    def calculate_AUCs(self):
        for concentration,repeats in self.concentration_maps.items():

            count = len(repeats)
            print('{} : Averaging {} repeats of {}'.format(self.isolate_name,count,concentration))

            reps = []
            for repID, rep in repeats.items():
                reps.append(rep)

            reps = np.asarray(reps)

            #Subtract blank
            reps = reps - self.blanks

            #Integrate numerically
            areas = np.trapz(reps, x=self.timepoints,axis=-1)

            areas_av = np.mean(areas)
            stderr = np.std(areas)/np.sqrt(len(reps))

            self.AUC_averages[concentration] = areas_av
            self.AUC_average_stderr[concentration] = stderr

    def plot_AUC(self,normalise=True,ax=None,**kwargs):

        n=len(self.AUC_averages)
        X = np.asarray([-1.0]*n,dtype='float')
        Y = np.asarray([-1.0]*n,dtype='float')
        Yerr = np.asarray([-1.0]*n,dtype='float')

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = ax

        if normalise:
            #Find minimum concentration condition
            minconc = np.min(np.asarray(list(self.AUC_averages.keys())))
            minarea = self.AUC_averages[minconc]
            print('{} : Normalising AUC plot to conc {} with area {}'.format(self.isolate_name,minconc,minarea))
        else:
            minarea = 1
            minconc = 'N/A'

        i=0
        for conc,area in self.AUC_averages.items():
            X[i] = conc
            Y[i] = area/minarea # normalise area
            Yerr[i] = self.AUC_average_stderr[conc]/minarea #and stderr
            i+=1


        ax.set_xlabel('Ciprofloxacin concentration (mg/L)')
        ax.set_ylabel('Normalised total growth')

        ax.errorbar(X,Y,yerr=Yerr,**kwargs)

        return [X,Y,Yerr]








def read_csv(path):
    frame = pandas.read_csv(path)
    names = list(frame.columns)

    assert 'time' in names

    blanks = []
    samples = {}

    for column in names:
        if column == 'time':
            timepoints = frame.loc[:,column]
            if timepoints[0] == 'time':
                times = timepoints[1:]
            else:
                raise ValueError('Double header not detected, check file.')


        else:
            [A,B,C] = column.split('_')

            #Check for blanks
            if A == 'Media' and B == 'blank':

                tseries_blank = frame.loc[:,column]
                tseries_blank = np.asarray(tseries_blank[1:],dtype='float')
                blanks.append(tseries_blank)

            elif A.startswith('L') and B in ['A','B','C','D'] and C in [str(conc) for conc in [16,8,4,2,1,0.5,0.1,0.01,0.001,0]]:
                if A not in samples:
                    tlapse = Timelapse(A)
                    tlapse.add_timepoints(np.asarray(times,dtype='float'))
                    samples[A] = tlapse

                tlapse = samples[A]
                tlapse.add_concentration(float(C))

                tseries = frame.loc[:,column]
                tseries = tseries[1:]

                tlapse.add_tseries(np.asarray(tseries,dtype='float'),repeat_ID=B,concentration=float(C))


    #Add blanks and calculate averages
    for code,sample in samples.items():
        sample.add_blanks(blanks)
        sample.calculate_averages()
        sample.calculate_AUCs()
    return samples



def d5PL_dP(p,x):
    '''return dF/dP(x) == vector of gradients wrt parameters at x'''

    a,b,c,d,g = p

    df_da = (1 + (x/c)**b)**(-g)
    df_db = -(a - d)*g*(x/c)**b*(1 + (x/c)**b)**(-1 - g)*np.log(x/c)
    df_dc = (b*(a - d)*g*(x/c)**b*(1 + (x/c)**b)**(-1 - g))/c
    df_dd = 1 - (1 + (x/c)**b)**(-g)
    df_dg = -(a - d)*(1 + (x/c)**b)**(-g)*np.log(1 + (x/c)**b)


    return [df_da,df_db,df_dc,df_dd,df_dg]



def dose_response_5PL(p,x):
    a,b,c,d,g = p
    return d + (a - d) / ((1 + (x / c) ** b) ** g)



def fit_dose_response(X=None,Y=None, sigmaY=None, n=None, f=None,df_dp=None,title=None, labels=None, MICs=None, fit_mapping=None,isolate_codes=None,timelapses=None):

    #colours = ['#5d8aa8','#318ce7','#0047ab'] #Legacy blue colours
    colours = ['#427BB6','#B6427B','#7BB642']
    markers = ['o','^','s']
    trace_count = len(Y)
    assert len(Y) == len(sigmaY) == len(labels) == len(MICs)


    fig1, (ax1, ax2) = plt.subplots(2,1,sharex=True,sharey=False,figsize=(6,8))

    X = np.asarray(X)
    idx = X.argsort() #Calculate sorting indices
    X = X[idx]

    for i in range(0,trace_count,1):
        colour = colours[i]
        marker = markers[i]

        Y_active=np.asarray(Y[i])
        sigmaY_active=np.asarray(sigmaY[i])
        deltaY_active=sigmaY_active/np.sqrt(n) #Compute standard error of the mean
        fit_decision = fit_mapping[i]
        MIC = MICs[i]

        #Sort in ascending order

        Y_active=Y_active[idx]
        sigmaY_active=sigmaY_active[idx]
        deltaY_active=deltaY_active[idx]

        ymid = Y_active.min()+(Y_active.max()-Y_active.min())/2
        mini = np.argmin((Y_active-ymid)**2)
        xmid= X[mini]

        p0 = np.asarray([0, 1, xmid, 1, 0.5])
        smallest_nonzero = np.min(X[np.nonzero(X)])
        xspace = np.linspace(smallest_nonzero, X.max(), 1000000)

        #Plot averages
        label = labels[i] if labels is not None else None
        ax1.errorbar(X,Y_active,yerr=deltaY_active, ecolor=colour, fmt = marker, color=colour, capsize=8, label=label)

        ax1.legend()
        ax1.axis(xmin=smallest_nonzero / 2,xmax=100, ymin=0.05,ymax=1.05 )
        ax1.set_xscale('log')
        #ax1.plot([MIC, MIC], [0, 1], color=colour, ls=':', lw=2)  # vertical line at MIC
        ax1.axvline(MIC,color=colour, ls=':', lw=2)
        ax2.axvline(MIC, color=colour, ls=':', lw=2)

        #ax2.plot([MIC, MIC], [0, 1], color=colour, ls=':', lw=2)  # vertical line at MIC
        #ax1.set_xlabel('CIP concentration (mg/L)')
        ax1.set_ylabel('Ratio of susceptible cells')
        ax1.set_title(title)

        #Show fits and bounds analysis if asked for

        if fit_decision:

            parinfo = [{'limits': (0,1)}, {'limits':(0,np.inf)}, {'limits': (0, X.max())}, {'limits':(0,1)}, {'limits':(0.1,3)}]
            fit = kmpfit.simplefit(f, p0, X, Y_active, err=sigmaY_active, parinfo=parinfo)

            derivative = df_dp(fit.params,xspace)
            yhat, upper, lower = fit.confidence_band(xspace, derivative, 0.95, f)

            print('Fitting ratio:')
            print("Best fit parameters for {}:".format(label), fit.params)
            print("Parameter errors:  :", fit.xerror)

            #Plot best fit
            ax1.plot(xspace,yhat, 'k--')

            # Plot bounds
            ax1.fill_between(xspace, lower, upper,
                             color='black', alpha=0.15)

        #Now plot AUC on second plot
        tlapse = timelapses[isolate_codes[i]]
        [X1,Y1,Yerr]=tlapse.plot_AUC(ax=ax2, ecolor=colour, fmt = marker, color=colour, capsize=8, label=label)

        tlapse.plot_averages()

        #Fit to areas if requested

        if fit_decision:
            p0 = np.asarray([0, -1, xmid, 1, 0.4])
            parinfo = [{'limits': (0, 1.0)}, {'limits': (-5,0)}, {'limits': (0, X.max())},
                       {'limits': (0, 1.0)}, {'limits': (0.1,0.4)}]
            fit = kmpfit.simplefit(f, p0, X1, Y1, err=Yerr*np.sqrt(3), parinfo=parinfo)
            derivative = df_dp(fit.params, xspace)
            yhat, upper, lower = fit.confidence_band(xspace, derivative, 0.95, f)
            print('Fitting area:')
            print("Best fit parameters for {}:".format(label), fit.params)
            print("Parameter errors:  :", fit.xerror)

            # Plot best fit
            ax2.plot(xspace, yhat, 'k--')

            # Plot bounds
            ax2.fill_between(xspace, lower, upper,
                             color='black', alpha=0.15)

    savepath = r'C:\Users\zagajewski\Desktop\fig6.svg'
    plt.savefig(savepath)
    plt.show()



if __name__ == "__main__":
    # sigmas are the standard deviations of the means, n is the number of repeats, Y is the mean. Copy-paste from origin

    Y_L36929 = [0.16667, 0.47333, 0.16667, 0.67333, 0.81333, 0.93, 0.97, 0.16333, 0.98667, 0.14667]  ####
    sigma_Y_L36929 = [0.03055, 0.23029, 0.03215, 0.21595, 0.11504, 0.05, 0.01, 0.05686, 0.00577, 0.04163]  ####

    Y_L48480 = [0.15667, 0.91667, 0.75, 0.94333, 0.91333, 0.92, 0.9, 0.33, 0.95667, 0.18]  ####
    sigma_Y_L48480 = [0.03215, 0.02082, 0.20224, 0.01528, 0.02517, 0.09539, 0.06928, 0.13077, 0.04041, 0.06083]  ####

    Y_L13834 = [0.08333, 0.09667, 0.11667, 0.07333, 0.08333, 0.09667, 0.17667, 0.07, 0.34333, 0.08]  ####
    sigma_Y_L13834 = [0.04041, 0.05508, 0.05132, 0.04163, 0.05859, 0.07371, 0.07506, 0.03606, 0.12583, 0.05568]  ####

    X = [0, 0.5, 0.1, 1, 2, 4, 8, 0.01, 16, 0.001]  ####
    n = 3

    # Path to csv
    csvpath = r'C:\Users\zagajewski\Desktop\Growth_curves_fig6_24hrs.csv'

    timelapses = read_csv(csvpath)
    fit_dose_response(X=X,Y=[Y_L48480,Y_L36929,Y_L13834],sigmaY=[sigma_Y_L48480,sigma_Y_L36929,sigma_Y_L13834],n=n, f=dose_response_5PL, df_dp=d5PL_dP, title=None, labels=['EC1','EC3','EC5'], isolate_codes=['L48480','L36929','L13834'], MICs=[0.008,0.5,72], fit_mapping=[True,True,False],timelapses=timelapses)