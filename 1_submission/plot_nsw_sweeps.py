import covasim as cv
import pandas as pd
import sciris as sc
import pylab as pl
import numpy as np
from matplotlib import ticker
import datetime as dt
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LogNorm

# Filepaths
resultsfolder = 'sweeps'
sensfolder = 'sweepssens'
figsfolder = 'figs'
process = False

# Parameter levels
T = sc.tic()
tlevels = [0.067, 0.1, 0.15, 0.19]
vlevels = np.arange(0, 5) / 4
mlevels = np.arange(0, 4) / 4
nt, nv, nm = len(tlevels), len(vlevels), len(mlevels)

# Fonts and sizes for all figures
font_size = 26
font_family = 'Proxima Nova'
pl.rcParams['font.size'] = font_size
pl.rcParams['font.family'] = font_family

################################################################################################
# Do processing if required
################################################################################################
if process:
    for thisfig in [resultsfolder,sensfolder]:
        results = {'cum_infections': {}, 'r_eff': {}, 'new_infections':{}, 'cum_quarantined':{}}
        for future_test_prob in tlevels:
            for name in ['cum_infections', 'r_eff', 'new_infections','cum_quarantined']: results[name][future_test_prob] = {}
            for venue_trace_prob in vlevels:
                for name in ['cum_infections', 'r_eff', 'new_infections','cum_quarantined']: results[name][future_test_prob][venue_trace_prob] = []
                for mask_uptake in mlevels:
                    print(f'mask_uptake: {mask_uptake}, venue_trace_prob: {venue_trace_prob}, future_test_prob: {future_test_prob}')
                    msim = sc.loadobj(f'{thisfig}/nsw_tracingsweeps_T{int(future_test_prob * 100)}_M{int(mask_uptake * 100)}_V{int(venue_trace_prob * 100)}.obj')
                    results['cum_quarantined'][future_test_prob][venue_trace_prob].append(msim.results['cum_quarantined'].values[-1]-msim.results['cum_quarantined'].values[244])
                    results['cum_infections'][future_test_prob][venue_trace_prob].append(msim.results['cum_infections'].values[-1]-msim.results['cum_infections'].values[244])
                    results['r_eff'][future_test_prob][venue_trace_prob].append(msim.results['r_eff'].values[-1])
                    results['new_infections'][future_test_prob][venue_trace_prob].append(msim.results['new_infections'].values)
        sc.saveobj(f'{thisfig}/nsw_sweep_results.obj', results)
#else:
#    results = sc.loadobj(f'{resultsfolder}/nsw_sweep_results.obj')



################################################################################################################
# Figure 2 and S2: grids of new infections
################################################################################################################

for thisfig in [resultsfolder, sensfolder]:

    # Fonts and sizes
    fig = pl.figure(figsize=(24,16))
    results = sc.loadobj(f'{thisfig}/nsw_sweep_results.obj')

    # Subplot sizes
    xgapl = 0.05
    xgapm = 0.017
    xgapr = 0.05
    ygapb = 0.05
    ygapm = 0.017
    ygapt = 0.05
    nrows = nt
    ncols = nv
    dx = (1-(ncols-1)*xgapm-xgapl-xgapr)/ncols
    dy = (1-(nrows-1)*ygapm-ygapb-ygapt)/nrows
    nplots = nrows*ncols
    ax = {}
    colors = pl.cm.GnBu(np.array([0.4,0.6,0.8,1.]))
    labels = ['0% masks', '25% masks', '50% masks', '75% masks']

    epsx = 0.003
    epsy = 0.008
    llpad = 0.01
    rlpad = 0.005

    if thisfig==resultsfolder:
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*0+epsy, '      90% testing      ', rotation=90, fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*1+epsy, '      80% testing      ', rotation=90, fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*2+epsy, '      65% testing      ', rotation=90, fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*3+epsy, '      50% testing      ', rotation=90, fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})

    elif thisfig==sensfolder:
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*0+epsy, '  90% symp. testing \n 60% contact testing ', rotation=90, fontsize=26, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*1+epsy, '  80% symp. testing \n 50% contact testing ', rotation=90, fontsize=26, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*2+epsy, '  65% symp. testing \n 40% contact testing ', rotation=90, fontsize=26, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
        pl.figtext(xgapl+dx*nv+xgapm*(nv-1)+rlpad, ygapb+(ygapm+dy)*3+epsy, '  50% symp. testing \n 30% contact testing ', rotation=90, fontsize=26, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})

    pl.figtext(xgapl+(dx+xgapm)*0+epsx, ygapb+dy*nm+ygapm*(nm-1)+llpad, '        0% tracing           ', fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
    pl.figtext(xgapl+(dx+xgapm)*1+epsx, ygapb+dy*nm+ygapm*(nm-1)+llpad, '        25% tracing          ', fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
    pl.figtext(xgapl+(dx+xgapm)*2+epsx, ygapb+dy*nm+ygapm*(nm-1)+llpad, '        50% tracing         ', fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
    pl.figtext(xgapl+(dx+xgapm)*3+epsx, ygapb+dy*nm+ygapm*(nm-1)+llpad, '        75% tracing          ', fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
    pl.figtext(xgapl+(dx+xgapm)*4+epsx, ygapb+dy*nm+ygapm*(nm-1)+llpad, '        100% tracing        ', fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})

    # Extract plot values
    def plinf(pn, what='new_infections'):
        # Get series for this plot number
        t = list(results['new_infections'].keys())[(nplots-1-pn)//nv]
        v = list(results['new_infections'][t].keys())[pn%nv]
        if what =='new_infections':
            #return np.array(([results['new_infections'][t][v][mm][214:] for mm in range(nm)]))
            return np.array([[results['new_infections'][t][v][mm][200+i:214+i].sum() / 14 for i in range(306-214)] for mm in range(nm)])
        elif what == 'cum_infections':
            return results['cum_infections'][t][v]

    @ticker.FuncFormatter
    def date_formatter(x, pos):
        return (cv.date('2020-09-30') + dt.timedelta(days=x)).strftime('%d-%b')

    for pn in range(nplots):
        ax[pn] = pl.axes([xgapl+(dx+xgapm)*(pn%ncols), ygapb+(ygapm+dy)*(pn//ncols), dx, dy])
        data = plinf(pn)
        for mi,mval in enumerate(mlevels):
            ax[pn].plot(range(len(data[mi,:])), data[mi,:], '-', lw=4, c=colors[mi], label=labels[mi], alpha=1.0)
            val = sc.sigfig(plinf(pn, what='cum_infections')[mi],3) if plinf(pn, what='cum_infections')[mi]<100 else sc.sigfig(plinf(pn, what='cum_infections')[mi],2)
            ax[pn].text(0.1, 180-mi*15, val.rjust(6), fontsize=20, family='monospace', color=colors[mi])

        ax[pn].set_ylim(0, 200)
        ax[pn].xaxis.set_major_formatter(date_formatter)

        if pn==4: pl.legend(loc='upper right', frameon=False, fontsize=20)
        if pn not in [0,5,10,15]:
            ax[pn].set_yticklabels([])
        else:
            ax[pn].set_ylabel('New infections')
        if pn not in range(nv):
            ax[pn].set_xticklabels([])
        else:
            xmin, xmax = ax[pn].get_xlim()
            ax[pn].set_xticks(pl.arange(xmin+5, xmax, 40))

    if thisfig==resultsfolder: figname = figsfolder+'/fig2_grid.png'
    elif thisfig==sensfolder: figname = figsfolder+'/figS2_grid.png'

    cv.savefig(figname, dpi=100)


#d = {'testing': [0.067]*nv*nm+[0.1]*nv*nm+[0.15]*nv*nm+[0.19]*nv*nm, 'tracing': [0.0]*nm+[0.25]*nm+[0.5]*nm+[0.75]*nm+[1.0]*nm+[0.0]*nm+[0.25]*nm+[0.5]*nm+[0.75]*nm+[1.0]*nm+[0.0]*nm+[0.25]*nm+[0.5]*nm+[0.75]*nm+[1.0]*nm+[0.0]*nm+[0.25]*nm+[0.5]*nm+[0.75]*nm+[1.0]*nm, 'masks': [0.0,0.25,0.5,0.75]*nt*nv}
#d['val'] = []
#for t in tlevels:
#    for v in vlevels:
#        d['val'].extend(sc.sigfig(results['cum_infections'][t][v],3))
#import pandas as pd
#df = pd.DataFrame(d)
#df.to_excel('sweepresults.xlsx')


################################################################################################################
# Figure 3: bar plot of cumulative infections
################################################################################################################
mainres = sc.loadobj(f'{resultsfolder}/nsw_sweep_results.obj')
sensres = sc.loadobj(f'{sensfolder}/nsw_sweep_results.obj')

# Subplot sizes
xgapl = 0.07
xgapm = 0.02
xgapr = 0.02
ygapb = 0.1
ygapm = 0.02
ygapt = 0.08
nrows = 1
ncols = 2
dx = (1-(ncols-1)*xgapm-xgapl-xgapr)/ncols
dy = (1-(nrows-1)*ygapm-ygapb-ygapt)/nrows
nplots = nrows*ncols
ax = {}
colors = pl.cm.GnBu(np.array([0.4,0.6,0.8,1.]))
mlabels = ['0% masks', '25% masks', '50% masks', '75% masks']
tlabels = ['50%', '65%', '80%', '90%']

fig = pl.figure(figsize=(24,8*nrows))

x = np.arange(len(tlabels))
width = 0.2  # the width of the bars

# Extract data
datatoplot = {}

datatoplot[0] = np.array([[mainres['cum_infections'][t][1.0][mi] for t in tlevels] for mi in range(nm)])
datatoplot[1] = np.array([[sensres['cum_infections'][t][1.0][mi] for t in tlevels] for mi in range(nm)])
#datatoplot[2] = np.array([[mainres['cum_quarantined'][t][1.0][mi] for t in tlevels] for mi in range(nm)])
#datatoplot[3] = np.array([[sensres['cum_quarantined'][t][1.0][mi] for t in tlevels] for mi in range(nm)])

# Headings
pl.figtext(xgapl+0.001, ygapb+dy*nrows+ygapm*(nrows-1)+0.01, '     Asymptomatic testing equal to symptomatic testing         ',
           fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})
pl.figtext(xgapl+xgapm+dx+0.001, ygapb+dy*nrows+ygapm*(nrows-1)+0.01, '     Asymptomatic testing lower than symptomatic testing      ',
           fontsize=30, fontweight='bold', bbox={'edgecolor':'none', 'facecolor':'silver', 'alpha':0.5, 'pad':4})

# Make plots
for pn in range(nplots):
    ax[pn] = pl.axes([xgapl+(dx+xgapm)*(pn%ncols), ygapb+(ygapm+dy)*(pn//ncols), dx, dy])
    data = datatoplot[pn]
    for mi,mval in enumerate(mlevels):
        ax[pn].bar(x+width*(mval*4-1.5), data[mi,:], width, color=colors[mi], label=mlabels[mi], alpha=1.0)

    ax[pn].set_xticks(x)
    ax[pn].set_xticklabels(tlabels)
    if pn <2:
        ax[pn].set_ylim(0, 20e3)
        ax[pn].set_xlabel('Symptomatic testing rate')
    else:
        ax[pn].set_ylim(0, 250e3)
    sc.boxoff()

    if pn in [0,2]:
        ax[pn].set_ylabel('Cumulative infections')
    if pn==1:
        pl.legend(loc='upper right', frameon=False, fontsize=20)
        ax[pn].set_yticklabels([])

cv.savefig(f'{figsfolder}/fig3_bars.png', dpi=100)


################################################################################################################
# Figure X: trade-off heatmaps
################################################################################################################

# Subplot sizes
xgapl = 0.07
xgapm = 0.02
xgapr = 0.02
ygapb = 0.3
ygapm = 0.02
ygapt = 0.08
nrows = 1
ncols = nv
dx = (1-(ncols-1)*xgapm-xgapl-xgapr)/ncols
dy = (1-(nrows-1)*ygapm-ygapb-ygapt)/nrows
nplots = nrows*ncols
ax = {}

# Create figure
fig = pl.figure(figsize=(24,10))

colors = pl.cm.GnBu(np.array([0.4,0.6,0.8,1.]))
mlabels = ['0% masks', '25% masks', '50% masks', '75% masks']
tlabels = ['50%', '65%', '80%', '90%']
vlabels = ['0% tracing', '25% tracing', '50% tracing', '75% tracing', '100% tracing']

M, T = np.meshgrid(mlevels, tlevels)
mt = M.reshape(nt*nm,)
tt = T.reshape(nt*nm,)
cmin, cmax = 0., 5.
lev_exp = np.arange(0., 5.1, 0.1)
levs = np.power(10, lev_exp)

# Load objects
for pn,vl in enumerate(vlevels):

    # Load in scenario multisims
    zi1 = np.array([mainres['cum_infections'][ti][vl] for ti in tlevels])
    z = zi1.reshape(nt*nm,)

    # Set axis and plot
    ax[pn] = pl.axes([xgapl+(dx+xgapm)*(pn%ncols), ygapb+(ygapm+dy)*(pn//ncols), dx, dy])
    im = ax[pn].imshow(zi1, cmap='Oranges')

    # Annotate
    for i in range(nm):
        for j in range(nt):
            c = sc.sigfig(zi1[j, i],3)
            ax[pn].text(i, j, str(c), va='center', ha='center')

    # Axis and plot labelling
    if pn == 0:
        ax[pn].set_ylabel('Symptomatic testing rate', fontsize=24, labelpad=20)
    ax[pn].set_xlabel('Mask uptake')
    ax[pn].set_title(vlabels[pn])

    ax[pn*100] = pl.axes([xgapl+(dx+xgapm)*(pn%ncols), 0.05, dx, 0.1])
    cbar = pl.colorbar(im, cax=ax[pn*100])
cv.savefig(f'{figsfolder}/figX_heatmaps.png', dpi=100)




sc.toc(T)





