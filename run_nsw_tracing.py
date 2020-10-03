import covasim as cv
import pandas as pd
import sciris as sc
import numpy as np

today = '2020-09-30'
tomorrow = '2020-10-01'
runtil = '2020-12-31'

def make_ints(make_future_ints=True, mask_uptake=None, venue_trace_prob=None, future_test_prob=None, mask_eff=0.3):
    # Make historical interventions
    initresponse = '2020-03-15'
    lockdown = '2020-03-23'
    reopen1  = '2020-05-01' # Two adults allowed to visit a house
    reopen2  = '2020-05-15' # Up to 5 adults can visit a house; food service and non-essential retail start to resume
    reopen3  = '2020-06-01' # Pubs and regional travel open, plus more social activities
    reopen4  = '2020-07-01' # large events, cinemas, museums, open; fewer restrictions on cafes/pubs/etc,
    school_dates = ['2020-05-11', '2020-05-18', '2020-05-25']
    comm_beta_aug = 0.7

    ints = [cv.clip_edges(days=[initresponse,lockdown]+school_dates, changes=[0.75, 0.05, 0.8, 0.9, 1.0], layers=['S'], do_plot=False),
                 cv.clip_edges(days=[lockdown, reopen2, reopen3, reopen4], changes=[0.5, 0.65, 0.75, 0.85], layers=['W'], do_plot=False),
                 cv.clip_edges(days=[lockdown, reopen2, reopen4], changes=[0, 0.5, 1.0], layers=['pSport'], do_plot=False),
                 cv.clip_edges(days=[lockdown, '2020-06-22'], changes=[0, 1.0], layers=['cSport'], do_plot=False),

                 cv.change_beta(days=[lockdown, reopen2, reopen4], changes=[1.2, 1.1, 1.], layers=['H'], do_plot=True),

                 cv.change_beta(days=[lockdown, reopen2], changes=[0, comm_beta_aug], layers=['church'], do_plot=False),
                 cv.change_beta(days=[lockdown, reopen1, reopen2, reopen3, reopen4], changes=[0.0, 0.3, 0.4, 0.5, comm_beta_aug], layers=['social'], do_plot=False),

                 # Dynamic layers ['C', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport', 'public_parks', 'large_events']
                 cv.change_beta(days=[lockdown], changes=[comm_beta_aug], layers=['C'], do_plot=True),
                 cv.change_beta(days=[lockdown, reopen4], changes=[0, comm_beta_aug], layers=['entertainment'], do_plot=False),
                 cv.change_beta(days=[lockdown, reopen2], changes=[0, comm_beta_aug], layers=['cafe_restaurant'], do_plot=False),
                 cv.change_beta(days=[lockdown, reopen3, reopen4], changes=[0, 0.5, comm_beta_aug], layers=['pub_bar'], do_plot=False),
                 cv.change_beta(days=[lockdown, reopen2, reopen4], changes=[0.2, 0.3, comm_beta_aug], layers=['transport'], do_plot=False),
                 cv.change_beta(days=[lockdown, reopen2, reopen4], changes=[0.4, 0.5, comm_beta_aug], layers=['public_parks'], do_plot=False),
                 cv.change_beta(days=[lockdown, reopen4], changes=[0.0, comm_beta_aug], layers=['large_events'], do_plot=False),
                 ]

    # Approximate a mask intervention by changing beta in all layers where people would wear masks - assuming not in schools, sport, social gatherings, or home
    mask_uptake_aug = 0.15
    mask_uptake_sep = 0.3
    mask_beta_change_aug = (1-mask_uptake_aug)*comm_beta_aug + mask_uptake_aug*mask_eff*comm_beta_aug
    mask_beta_change_sep = (1-mask_uptake_sep)*comm_beta_aug + mask_uptake_sep*mask_eff*comm_beta_aug
    ints += [cv.change_beta(days=['2020-08-01', '2020-08-31'] * 8, changes=[mask_beta_change_aug, 0.7] * 8,
                                 layers=['church', 'C', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport',
                                         'public_parks', 'large_events']),
             cv.change_beta(days=['2020-09-01', today] * 8, changes=[mask_beta_change_sep, 0.7] * 8,
                            layers=['church', 'C', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport',
                                    'public_parks', 'large_events'])
             ]

    if make_future_ints:
        ints += [cv.change_beta(days=[today] * 8, changes=[(1 - mask_uptake) * comm_beta_aug + mask_uptake * mask_eff * comm_beta_aug] * 8,
                       layers=['church', 'C', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport',
                               'public_parks', 'large_events'])]

    # Testing
    symp_prob_prelockdown = 0.04  # Limited testing pre lockdown
    symp_prob_lockdown = 0.07  # 0.065 #Increased testing during lockdown
    symp_prob_postlockdown = 0.19 # 0.165 # Testing since lockdown
    asymp_quar_prob_postlockdown = (1-(1-symp_prob_postlockdown)**10)
    future_asymp_test_prob = (1-(1-future_test_prob)**10)/2

    ints += [cv.test_prob(start_day=0, end_day=lockdown, symp_prob=symp_prob_prelockdown, asymp_quar_prob=0.01, do_plot=False),
             cv.test_prob(start_day=lockdown, end_day=reopen2, symp_prob=symp_prob_lockdown, asymp_quar_prob=0.01,do_plot=False),
             cv.test_prob(start_day=reopen2, end_day=today, symp_prob=symp_prob_postlockdown, asymp_quar_prob=asymp_quar_prob_postlockdown,do_plot=True)]

    if make_future_ints:
        ints += [cv.test_prob(start_day=tomorrow, symp_prob=future_test_prob, asymp_quar_prob=future_asymp_test_prob, do_plot=True)]

    # Tracing
    trace_probs = {'H': 1, 'S': 0.95, # Home and school
                   'W': 0.8, 'pSport': 0.8, 'cSport': 0.8, 'social': 0.8, # Work and social
                   'C': 0.05, 'public_parks': 0.05, # Non-venue-based
                   'church': 0.5, 'entertainment': 0.5, 'cafe_restaurant': 0.5, 'pub_bar': 0.5, 'large_events': 0.5, # Venue-based
                   'transport': 0.1, # Transport
                   }
    trace_time  = {'H': 0, 'S': 0.5, # Home and school
                   'W': 1, 'pSport': 1, 'cSport': 1, 'social': 1, # Work and social: [0.29, 0.59, 0.74, 0.78, 0.80, 0.80, 0.80, 0.80]
                   'C': 2, 'public_parks': 2, # Non-venue-based: [0.0068, 0.0203, 0.0338, 0.0429, 0.0474, 0.0492, 0.0498, 0.0499]
                   'church': 1, 'entertainment': 1, 'cafe_restaurant': 1, 'pub_bar': 1, 'large_events': 1, # Venue-based: [0.068, 0.203, 0.338, 0.429, 0.474, 0.492, 0.498, 0.499]
                   'transport': 1, # Transport: [0.014, 0.041, 0.068, 0.086, 0.095, 0.098, 0.100, 0.100]
                   }
    trace_time_f  = {'H': 0, 'S': 0.5, # Home and school
                   'W': 1, 'pSport': 1, 'cSport': 1, 'social': 1, # Work and social: [0.29, 0.59, 0.74, 0.78, 0.80, 0.80, 0.80, 0.80]
                   'C': 2, 'public_parks': 2, # Non-venue-based: [0.0068, 0.0203, 0.0338, 0.0429, 0.0474, 0.0492, 0.0498, 0.0499]
                   'church': 1, 'entertainment': 1, 'cafe_restaurant': 1, 'pub_bar': 1, 'large_events': 1, # Venue-based: [0.068, 0.203, 0.338, 0.429, 0.474, 0.492, 0.498, 0.499]
                   'transport': 2, # Transport: [0.014, 0.041, 0.068, 0.086, 0.095, 0.098, 0.100, 0.100]
                   }
    ints += [cv.contact_tracing(trace_probs=trace_probs, trace_time=trace_time, distribute_times=True, start_day=0, end_day=today, do_plot=False)]

    if make_future_ints:
        ints += [cv.contact_tracing(trace_probs={'H': 1, 'S': 0.95, 'W': 0.8, 'pSport': 0.8, 'cSport': 0.8, 'social': 0.8,
                                                 'C': 0.05, 'public_parks': 0.05,
                                                 'church': venue_trace_prob, 'entertainment': venue_trace_prob, 'cafe_restaurant': venue_trace_prob, 'pub_bar': venue_trace_prob, 'large_events': venue_trace_prob, 'transport': 0.1},
                                    trace_time=trace_time_f, distribute_times=True, start_day=tomorrow, do_plot=False)]

    # Close borders, then open them again to account for Victorian imports and leaky quarantine
    ints += [cv.dynamic_pars({'n_imports': {'days': [14, 112, 116], 'vals': [0, 8, 0]}}, do_plot=False)]

    return ints


def make_sim(do_make_ints=True, make_future_ints=True, mask_uptake=None, venue_trace_prob=None, future_test_prob=None, mask_eff=0.3, load_pop=True, popfile='nswppl.pop', datafile=None):

    layers = ['H', 'S', 'W', 'C', 'church', 'pSport', 'cSport', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport', 'public_parks', 'large_events', 'social']

    end_day = runtil

    pars = {'pop_size': 100e3,
            'pop_infected': 110,
            'pop_scale': 75,
            'rescale': True,
            'rand_seed': 1,
            'beta': 0.0247, #0.024595,#0.02465,  Overall beta to use for calibration [0.0242, 0.02445, 0.02462, 0.02447 GOOD, 0.02437 GOOD] (0.02467,0.02456 maybe with higher sep masks), 0.02446 close
                                    # H     S       W       C       church  psport  csport  ent     cafe    pub     trans   park    event   soc
            'contacts':    pd.Series([4,    21,     5,      1,      20,     40,     30,     25,     19,     30,     25,     10,     50,     6], index=layers).to_dict(),
            'beta_layer':  pd.Series([1,    0.25,   0.3,    0.1,    0.04,   0.2,    0.1,    0.01,   0.04,   0.06,   0.16,   0.03,   0.01,   0.3], index=layers).to_dict(),
            'iso_factor':  pd.Series([0.2,  0,      0,      0.1,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0], index=layers).to_dict(),
            'quar_factor': pd.Series([1,    0.1,    0.1,    0.2,    0.01,   0,      0,      0,      0,      0,      0.1 ,   0,      0,      0], index=layers).to_dict(),
            'n_imports': 2, # Number of new cases to import per day -- varied over time as part of the interventions
            'start_day': '2020-03-01',
            'end_day': end_day,
            'verbose': .1}

    sim = cv.Sim(pars=pars,
                 datafile=datafile,
                 popfile=popfile,
                 load_pop=load_pop)

    if do_make_ints: sim.pars['interventions'] = make_ints(mask_eff=mask_eff, make_future_ints=make_future_ints, mask_uptake=mask_uptake, venue_trace_prob=venue_trace_prob, future_test_prob=future_test_prob)
    #sim.initialize()

    return sim


# Start setting up to run
# NB, this file assumes that you've already generated a population file saved in the same folder as this script, called nswpop.pop

T = sc.tic()

# Settings
whattorun = ['calibration', 'tracingsweeps', 'maskscenarios'][0]
domulti = True
doplot = True
dosave = True
n_runs = 10

# Filepaths
resultsfolder = 'calibration'
figsfolder = 'figs'
datafile = 'nsw_epi_data_os_removed.csv'

# Plot settings
to_plot = sc.objdict({
    'Cumulative diagnoses': ['cum_diagnoses'],
    'Cumulative deaths': ['cum_deaths'],
    'Cumulative infections': ['cum_infections'],
    'New infections': ['new_infections'],
    'Daily diagnoses': ['new_diagnoses'],
    'Active infections': ['n_exposed']
    })

# Make sim for calibration
if whattorun=='calibration':

    s0 = make_sim(do_make_ints=True, mask_uptake=0.3, venue_trace_prob=0.5, future_test_prob=0.9, mask_eff=0.3, load_pop=True,
             popfile='nswppl.pop', datafile=datafile)

    if domulti:
        msim = cv.MultiSim(base_sim=s0)
        msim.run(n_runs=n_runs)
        msim.reduce()
        if dosave: msim.save(f'nsw_{whattorun}.obj')
        if doplot:
            msim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'{figsfolder}/nsw_{whattorun}.png',
                  legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=21)
    else:
        s0.run(until=today)
        s0.finalize()  # Finalize the results
        if dosave: s0.save(f'{resultsfolder}/nsw_{whattorun}_single.obj')
        if doplot:
            s0.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'{figsfolder}/nsw_{whattorun}.png',
                  legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)



if whattorun=='tracingsweeps':

    res_to_keep = ['cum_infections', 'new_infections', 'cum_quarantined']

    results = {k:{} for k in res_to_keep}
    labels = []

    for future_test_prob in [0.19]: #[0.067, 0.1, 0.15, 0.19]

        for name in res_to_keep: results[name][future_test_prob] = {}
        for venue_trace_prob in np.arange(0, 5) / 4:
            for name in res_to_keep: results[name][future_test_prob][venue_trace_prob] = []
            for mask_uptake in np.arange(0, 4) / 4:

                # Make original sim
                s0 = make_sim(do_make_ints=True, make_future_ints=True, mask_uptake=mask_uptake, venue_trace_prob=venue_trace_prob, future_test_prob=future_test_prob, mask_eff=0.3, load_pop=True,
                              popfile='nswppl.pop', datafile=datafile)
                s0.run(until=today)
                print(f'mask_uptake: {mask_uptake}, venue_trace_prob: {venue_trace_prob}, future_test_prob: {future_test_prob}')

                # Copy sims
                sims = []
                for seed in range(n_runs):
                    sim = s0.copy()
                    sim.label = "T" + str(future_test_prob) + "_M" + str(mask_uptake) + "_V" + str(venue_trace_prob)
                    sim['rand_seed'] = seed
                    sim.set_seed()
                    sims.append(sim)

                msim = cv.MultiSim(sims)
                msim.run()
                msim.reduce()
                if dosave:
                    msim.save(f'{resultsfolder}/nsw_{whattorun}_T{int(future_test_prob*100)}_M{int(mask_uptake*100)}_V{int(venue_trace_prob*100)}.obj')
                results['cum_infections'][future_test_prob][venue_trace_prob].append(msim.results['cum_infections'].values[-1]-msim.results['cum_infections'].values[214])
                results['cum_quarantined'][future_test_prob][venue_trace_prob].append(msim.results['cum_quarantined'].values[-1]-msim.results['cum_quarantined'].values[214])
                results['new_infections'][future_test_prob][venue_trace_prob].append(msim.results['new_infections'].values)
    if dosave:
        sc.saveobj(f'{resultsfolder}/nsw_sweep_results.obj', results)



if whattorun=='maskscenarios':

    mask_beta_change = [0.55,0.59,0.60,0.62,0.70]
    all_layer_counts = {}
    layer_remap = {'H': 0, 'S': 1, 'W': 2, 'church': 3, 'pSport': 3, 'cSport': 3, 'social': 3, 'C': 4,
                   'entertainment': 4,
                   'cafe_restaurant': 4, 'pub_bar': 4, 'transport': 4, 'public_parks': 4, 'large_events': 4,
                   'importation': 4, 'seed_infection': 4}
    n_new_layers = 5  # H, S, W, DC, SC

    for jb in mask_beta_change:

        # Make original sim
        s0 = make_sim(mask_beta_change=jb, load_pop=True, popfile='nswppl.pop', datafile=datafile)
        s0.run(until=today)

        # Copy sims
        sims = []
        for seed in range(n_runs):
            sim = s0.copy()
            sim['rand_seed'] = seed
            sim.set_seed()
            sims.append(sim)
        msim = cv.MultiSim(sims)
        msim.run(n_runs=n_runs, reseed=True, noise=0, keep_people=True)

        all_layer_counts[jb] = np.zeros((n_runs, s0.npts, n_new_layers))

        for sn, sim in enumerate(msim.sims):
            tt = sim.make_transtree()
            for source_ind, target_ind in tt.transmissions:
                dd = tt.detailed[target_ind]
                date = dd['date']
                layer_num = layer_remap[dd['layer']]
                all_layer_counts[jb][sn, date, layer_num] += sim.rescale_vec[date]

        if dosave: msim.save(f'{resultsfolder}/nsw_{whattorun}_{int(jb*100)}.obj')
        if dosave: sc.saveobj(f'{resultsfolder}/nsw_layer_counts.obj', all_layer_counts)
        if doplot:
            msim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'{figsfolder}/nsw_{whattorun}_{int(jb*100)}.png',
              legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=21)


sc.toc(T)


