import datetime as dt
from collections import defaultdict

import covasim as cv
import covasim.defaults as cvd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pylab as pl
from covasim import utils as cvu

import sciris as sc


class PolicySchedule(cv.Intervention):
    def __init__(self, baseline: dict, policies: dict):
        """
        Create policy schedule

        The policies passed in represent all of the possible policies that a user
        can subsequently schedule using the methods of this class

        Example usage:

            baseline = {'H':1, 'S':0.75}
            policies = {}
            policies['Close schools'] = {'S':0}
            schedule = PolicySchedule(baseline, policies)
            schedule.add('Close schools', 10) # Close schools on day 10
            schedule.end('Close schools', 20) # Reopen schools on day 20
            schedule.remove('Close schools')  # Don't use the policy at all

        Args:
            baseline: Baseline (relative) beta layer values e.g. {'H':1, 'S':0.75}
            policies: Dict of policies containing the policy name and relative betas for each policy e.g. {policy_name: {'H':1, 'S':0.75}}

        """
        super().__init__()
        self._baseline = baseline  #: Store baseline relative betas for each layer
        self.policies = sc.dcp(policies)  #: Store available policy interventions (name and effect)
        for policy in self.policies:
            self.policies[policy] = {k:v for k,v in self.policies[policy].items() if not pd.isna(v)}
            assert set(self.policies[policy].keys()).issubset(self._baseline.keys()), f'Policy "{policy}" has effects on layers not included in the baseline'
        self.policy_schedule = []  #: Store the scheduling of policies [(start_day, end_day, policy_name)]
        self.days = {}  #: Internal cache for when the beta_layer values need to be recalculated during simulation. Updated using `_update_days`

    def start(self, policy_name: str, start_day: int) -> None:
        """
        Change policy start date

        If the policy is not already present, then it will be added with no end date

        Args:
            policy_name: Name of the policy to change start date for
            start_day: Day number to start policy

        Returns: None

        """
        n_entries = len([x for x in self.policy_schedule if x[2] == policy_name])
        if n_entries < 1:
            self.add(policy_name, start_day)
            return
        elif n_entries > 1:
            raise Exception('start_policy() cannot be used to start a policy that appears more than once - need to manually add an end day to the desired instance')

        for entry in self.policy_schedule:
            if entry[2] == policy_name:
                entry[0] = start_day

        self._update_days()

    def end(self, policy_name: str, end_day: int) -> None:
        """
        Change policy end date

        This only works if the policy only appears once in the schedule. If a policy gets used multiple times,
        either add the end days upfront, or insert them directly into the policy schedule. The policy should
        already appear in the schedule

        Args:
            policy_name: Name of the policy to end
            end_day: Day number to end policy (policy will have no effect on this day)

        Returns: None

        """

        n_entries = len([x for x in self.policy_schedule if x[2] == policy_name])
        if n_entries < 1:
            raise Exception('Cannot end a policy that is not already scheduled')
        elif n_entries > 1:
            raise Exception('end_policy() cannot be used to end a policy that appears more than once - need to manually add an end day to the desired instance')

        for entry in self.policy_schedule:
            if entry[2] == policy_name:
                if end_day <= entry[0]:
                    raise Exception(f"Policy '{policy_name}' starts on day {entry[0]} so the end day must be at least {entry[0]+1} (requested {end_day})")
                entry[1] = end_day

        self._update_days()

    def add(self, policy_name: str, start_day: int, end_day: int = np.inf) -> None:
        """
        Add a policy to the schedule

        Args:
            policy_name: Name of policy to add
            start_day: Day number to start policy
            end_day: Day number to end policy (policy will have no effect on this day)

        Returns: None

        """
        assert policy_name in self.policies, 'Unrecognized policy'
        self.policy_schedule.append([start_day, end_day, policy_name])
        self._update_days()

    def remove(self, policy_name: str) -> None:
        """
        Remove a policy from the schedule

        All instances of the named policy will be removed from the schedule

        Args:
            policy_name: Name of policy to remove

        Returns: None

        """

        self.policy_schedule = [x for x in self.policy_schedule if x[2] != policy_name]
        self._update_days()

    def _update_days(self) -> None:
        # This helper function updates the list of days on which policies start or stop
        # The apply() function only gets run on those days
        self.days = {x[0] for x in self.policy_schedule}.union({x[1] for x in self.policy_schedule if np.isfinite(x[1])})

    def _compute_beta_layer(self, t: int) -> dict:
        # Compute beta_layer at a given point in time
        # The computation is done from scratch each time
        beta_layer = self._baseline.copy()
        for start_day, end_day, policy_name in self.policy_schedule:
            rel_betas = self.policies[policy_name]
            if t >= start_day and t < end_day:
                for layer in beta_layer:
                    if layer in rel_betas:
                        beta_layer[layer] *= rel_betas[layer]
        return beta_layer

    def apply(self, sim: cv.BaseSim):
        if sim.t in self.days:
            sim['beta_layer'] = self._compute_beta_layer(sim.t)
            if sim['verbose']:
                print(f"PolicySchedule: Changing beta_layer values to {sim['beta_layer']}")
                for entry in self.policy_schedule:
                    if sim.t == entry[0]:
                        print(f'PolicySchedule: Turning on {entry[2]}')
                    elif sim.t == entry[1]:
                        print(f'PolicySchedule: Turning off {entry[2]}')

    def plot_gantt(self, max_time=None, start_date=None, interval=None, pretty_labels=None):
        """
        Plot policy schedule as Gantt chart

        Returns: A matplotlib figure with a Gantt chart

        """
        fig, ax = plt.subplots()
        if max_time:
            max_time += 5
        else:
            max_time = np.nanmax(np.array([x[1] for x in self.policy_schedule if np.isfinite(x[1])]))

        #end_dates = [x[1] for x in self.policy_schedule if np.isfinite(x[1])]
        if interval:
            xmin, xmax = ax.get_xlim()
            ax.set_xticks(pl.arange(xmin, xmax + 1, interval))

        if start_date:
            @ticker.FuncFormatter
            def date_formatter(x, pos):
                return (start_date + dt.timedelta(days=x)).strftime('%b-%d')

            ax.xaxis.set_major_formatter(date_formatter)
            if not interval:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel('Dates')
            ax.set_xlim((0, max_time + 5))  # Extend a few days so the ends of policies can be seen

        else:
            ax.set_xlim(0, max_time + 5)  # Extend a few days so the ends of policies can be seen
            ax.set_xlabel('Days')
        schedule = sc.dcp(self.policy_schedule)
        if pretty_labels:
            policy_index = {pretty_labels[x]: i for i, x in enumerate(self.policies.keys())}
            for p, pol in enumerate(schedule):
               pol[2] = pretty_labels[pol[2]]
            colors = sc.gridcolors(len(pretty_labels))
        else:
            policy_index = {x: i for i, x in enumerate(self.policies.keys())}
            colors = sc.gridcolors(len(self.policies))
        ax.set_yticks(np.arange(len(policy_index.keys())))
        ax.set_yticklabels(list(policy_index.keys()))
        ax.set_ylim(0 - 0.5, len(policy_index.keys()) - 0.5)

        for start_day, end_day, policy_name in schedule:
            if not np.isfinite(end_day):
                end_day = 1e6 # Arbitrarily large end day
            ax.broken_barh([(start_day, end_day - start_day)], (policy_index[policy_name] - 0.5, 1), color=colors[policy_index[policy_name]])

        return fig


class SeedInfection(cv.Intervention):
    """
    Seed a fixed number of infections

    This class facilities seeding a fixed number of infections on a per-day
    basis.

    Infections will only be seeded on specified days

    """

    def __init__(self, infections: dict):
        """

        Args:
            infections: Dictionary with {day_index:n_infections}

        """
        super().__init__()
        self.infections = infections  #: Dictionary mapping {day: n_infections}. Day can be an int, or a string date like '20200701'

    def initialize(self, sim):
        super().initialize(sim)
        self.infections = {sim.day(k): v for k, v in self.infections.items()}  # Convert any day strings to ints

    def apply(self, sim):
        if sim.t in self.infections:
            susceptible_inds = cvu.true(sim.people.susceptible)

            if len(susceptible_inds) < self.infections[sim.t]:
                raise Exception('Insufficient people available to infect')

            targets = cvu.choose(len(susceptible_inds), self.infections[sim.t])
            target_inds = susceptible_inds[targets]
            sim.people.infect(inds=target_inds)


class DynamicTrigger(cv.Intervention):
    """
    Execute callback during simulation execution
    """

    def __init__(self, condition, action, once_only=False):
        """
        Args:
            condition: A function `condition(sim)` function that returns True or False
            action: A function `action(sim)` that runs if the condition was true
            once_only: If True, the action will only execute once
        """
        super().__init__()
        self.condition = condition  #: Function that
        self.action = action
        self.once_only = once_only
        self._ran = False

    def apply(self, sim):
        """
        Check condition and execute callback
        """
        if not (self.once_only and self._ran) and self.condition(sim):
            self.action(sim)
            self._ran = True

class test_prob_quarantine(cv.test_prob):
    def __init__(self, quarantine_compliance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quarantine_compliance = quarantine_compliance  #: Compliance level for individuals in general population isolating after testing. People already in quarantine are assumed to be compliant

    def apply(self, sim):
        super().apply(sim)

        tested_today_inds = cvu.true(sim.people.date_tested==sim.t)
        if len(tested_today_inds):
            # If people are meant to quarantine while waiting for their test, then quarantine some/all of the people waiting for tests
            quar_inds = cvu.binomial_filter(self.quarantine_compliance, tested_today_inds)
            sim.people.schedule_quarantine(quar_inds, period=self.test_delay)

class test_total_prob_with_quarantine(cv.test_prob):
    """
    Testing based on probability with quarantine during tests
    """

    def __init__(self, *args, swab_delay, test_isolation_compliance, leaving_quar_prob=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.swab_delay = swab_delay
        self.test_isolation_compliance = test_isolation_compliance  #: Compliance level for individuals in general population isolating after testing. People already in quarantine are assumed to be compliant
        self.leaving_quar_prob = leaving_quar_prob  # Probability of testing for people leaving quarantine e.g. set to 1 to ensure people test before leaving quarantine

    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # TEST LOGIC
        # 1. People who become symptomatic in the general community will wait `swab_delay` days before getting tested, at rate `symp_prob`
        # 2. People who become symptomatic while in quarantine will test immediately at rate `symp_quar_test`
        # 3. People who are symptomatic and then are ordered to quarantine, will test immediately at rate `symp_quar_test`
        # 4. People who have severe symptoms will be tested
        # 5. People test before leaving quarantine at rate `leaving_quar_prob` (set to 1 to ensure everyone leaving quarantine must have been tested)
        # 6. People that have been diagnosed will not be tested
        # 7. People that are already waiting for a diagnosis will not be retested
        # 8. People quarantine while waiting for their diagnosis with compliance `test_isolation_compliance`
        # 9. People already on quarantine while tested will not have their quarantine shortened, but if they are tested at the end of their
        #    quarantine, the quarantine will be extended

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = self.asymp_prob*np.ones(sim.n)  # Begin by assigning equal testing probability to everyone equal to the asymptomatic rate

        # (1) People wait swab_delay days before they decide to start testing. If swab_delay is 0 then they will be eligible as soon as they are symptomatic
        symp_inds = cvu.true(sim.people.symptomatic)  # People who are symptomatic
        symp_test_inds = symp_inds[sim.people.date_symptomatic[symp_inds] == (t - self.swab_delay)]  # People who became symptomatic previously and are eligible to test today
        test_probs[symp_test_inds] = self.symp_prob

        # People whose symptomatic scheduled day falls during quarantine will test at the symp_quar_prob rate
        # People who are already symptomatic, missed their test, and then enter quarantine, will test at the symp_quar_prob rate
        # People get quarantined at 11:59pm so the people getting quarantined today haven't been quarantined yet.
        # The order is
        # Day 4 - Test, quarantine people waiting for results
        # Day 4 - Trace
        # Day 4 - Quarantine known contacts
        # Day 5 - Test, nobody has entered quarantine on day 5 yet - if someone was symptomatic and untested and was quarantined *yesterday* then
        #         they need to be tested *today*

        # Someone waiting for a test result shouldn't retest. So we check that they aren't already waiting for their test.
        # Note that people are diagnosed after interventions are executed,
        # therefore if someone is tested on day 3 and the test delay is 2, on day 5 then sim.people.diagnosed will NOT
        # be true at the point where this code is executing. Therefore, they should not be eligible to retest. It's
        # like they are going to receive their results at 11:59pm so the decisions they make during the day are based
        # on not having been diagnosed yet. Hence > is used here so that on day 3+2=5, they won't retest. (i.e. they are
        # waiting for their results if the day they recieve their results is > the current day). Note that they become
        # symptomatic prior to interventions e.g. they wake up with symptoms
        if sim.t > 0:
            # If quarantined, there's no swab delay

            # (2) People who become symptomatic while quarantining test immediately
            quarantine_test_inds = symp_inds[sim.people.quarantined[symp_inds] & (sim.people.date_symptomatic[symp_inds] == t)]  # People that became symptomatic today while already on quarantine
            test_probs[quarantine_test_inds] = self.symp_quar_prob  # People with symptoms in quarantine are eligible to test without waiting

            # (3) People who are symptomatic and undiagnosed before entering quarantine, test as soon as they are quarantined
            newly_quarantined_test_inds = cvu.true((sim.people.date_quarantined == (sim.t - 1)) & sim.people.symptomatic & ~sim.people.diagnosed)  # People that just entered quarantine, who are current symptomatic and undiagnosed
            test_probs[newly_quarantined_test_inds] = self.symp_quar_prob  # People with symptoms that just entered quarantine are eligible to test

        # (4) People with severe symptoms that would be hospitalised are guaranteed to be tested
        test_probs[sim.people.severe] = 1.0  # People with severe symptoms are guaranteed to be tested unless already diagnosed or awaiting results

        # (5) People leaving quarantine test before leaving
        # This tests policies for testing people at least once during quarantine
        # - If leaving_quar_prob=1 then everyone leaving quarantine must have been tested during quarantine
        # - If someone was tested during their quarantine, they don't need to test again
        if self.leaving_quar_prob:
            to_test = cvu.true(sim.people.quarantined)  # Everyone currently on quarantine
            to_test = to_test[(sim.people.date_end_quarantine[to_test] - self.test_delay) == sim.t]  # Everyone leaving quarantine that needs to have been tested by today at the latest
            to_test = to_test[~(sim.people.date_tested[to_test] > sim.people.date_quarantined[to_test])]  # Note that this is not the same as <= because of NaNs - if someone was never tested, then both <= and > are False
            test_probs[to_test] = np.maximum(test_probs[to_test], self.leaving_quar_prob)  # If they are already supposed to test at a higher rate e.g. severe symptoms, keep it

        # (6) People that have been diagnosed aren't tested
        diag_inds = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.0  # People who are diagnosed or awaiting test results don't test

        # (7) People waiting for results don't get tested
        tested_inds = cvu.true(np.isfinite(sim.people.date_tested))
        pending_result_inds = tested_inds[(sim.people.date_tested[tested_inds] + self.test_delay) > sim.t]  # People who have been tested and will receive test results after the current timestep
        test_probs[pending_result_inds] = 0.0  # People awaiting test results don't test

        # Test people based on their per-person test probability
        test_inds = cvu.true(cvu.binomial_arr(test_probs))
        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)  # Actually test people
        sim.results['new_tests'][t] += int(len(test_inds))

        if self.test_isolation_compliance:
            # If people are meant to quarantine while waiting for their test, then quarantine some/all of the people waiting for tests
            quar_inds = cvu.binomial_filter(self.test_isolation_compliance, test_inds)
            sim.people.schedule_quarantine(quar_inds, period=self.test_delay)

        return test_probs


class DynamicContactTracing(cv.Intervention):
    """
    Contact tracing including infection log

    If many layers are dynamic, then `cv.contact_tracing` may miss interactions resulting in infection
    because the contact layer had been regenerated in the meantime. Actually tracking every contact is
    prohibitively computationally expensive. But augmenting the current contacts in the layer with
    the infections that have been recorded is an efficient approximation that allows those infections
    to be identified.

    """

    def __init__(self, trace_probs:dict, trace_time:dict,quarantine_period=14,start_day=0, end_day=None):
        super().__init__()
        self.start_day = start_day
        self.end_day = end_day
        self.trace_probs = trace_probs
        self.trace_time = trace_time
        self.quarantine_period = quarantine_period

    def initialize(self, sim):
        super().initialize(sim)
        self.start_day = sim.day(self.start_day)
        self.end_day = sim.day(self.end_day)
        assert self.trace_time.keys() == self.trace_probs.keys(), 'Trace time and trace probability must both be specified for the same layers'

    def apply(self, sim):
        """
        Trace and notify contacts

        Tracing involves two steps

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        """
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        trace_inds = self.select_cases(sim)
        contacts = self.identify_contacts(sim, trace_inds)
        self.notify_contacts(sim, contacts)


    def select_cases(self, sim: cv.Sim) -> np.ndarray:
        """
        Return people to be traced at this time step

        Args:
            sim:

        Returns: Array of people indexes to contact

        """
        trace_inds = cvu.true(sim.people.date_diagnosed == sim.t).astype(np.int64)  # Diagnosed this time step
        return trace_inds

    def identify_contacts(self, sim: cv.Sim, trace_inds: np.ndarray) -> dict:
        """
        Return contacts to notify by trace time

        Args:
            sim:
            trace_inds: Ind

        Returns: {trace_time: inds}

        """

        contacts = defaultdict(set)
        ind_set = set(trace_inds)

        if not ind_set:
            return contacts

        # Filter the infection log to select just the infections involving the people being traced
        # This is bi-directional because if the source could identify the target as a contact, the
        # reverse is assumed to be true as well (i.e. if A infects B and then B is traced, B would be able
        # to identify A as a contact of theirs).
        actual_infections = [x for x in sim.people.infection_log if (x['source'] in ind_set or x['target'] in ind_set)]  # People who were infected in a traceable layer involving the person being traced

        # Extract the indices of the people who'll be contacted
        for lkey in self.trace_probs:
            trace_prob = self.trace_probs[lkey]

            if trace_prob == 0:
                continue

            # Find current layer contacts
            contact_set = cvu.find_contacts(sim.people.contacts[lkey]['p1'], sim.people.contacts[lkey]['p2'], trace_inds)

            # Add interactions at previous timesteps that resulted in transmission. It's bi-directional because if the source
            # interacts with the target, the target would be able to name the source as a known contact with the same probability
            # as in the reverse direction.
            for infection in actual_infections:
                if infection['layer'] == lkey:
                    contact_set.add(infection['source'])
                    contact_set.add(infection['target'])

            # Check contacts
            contact_set = contact_set.difference(ind_set)  # The people being traced may have been added to the contact_set because they could be contacts of each other, but they don't need to be notified, so remove them
            edge_inds = np.fromiter(contact_set, dtype=cvd.default_int)
            edge_inds.sort()  # Make sure the edges come out in a consistent order for reproducibility

            trace_time = self.trace_time[lkey]
            contacts[trace_time].update(cvu.binomial_filter(trace_prob, edge_inds))  # Filter the indices according to the probability of being able to trace this layer

        array_contacts = {}
        for trace_time, inds in contacts.items():
            array_contacts[trace_time] = np.fromiter(inds, dtype=cvd.default_int) # NB the order doesn't matter here because this gets used in a vector operation
        return array_contacts

    def notify_contacts(self, sim, contacts: dict):
        # Quarantine ends after `self.quarantine_period` days from today, regardless of the trace time
        # This slightly overestimates quarantine duration in cases where tracing was performed via the
        # infection log, because the contact would have occurred up to a few days earlier. This approximation
        # means that we don't need to set a per-contact trace time, however, and instead we can use
        # a per-layer trace time, which is more computationally efficient.
        #
        # This being the case, if someone is traced in multiple layers with different trace times, they will
        # always be scheduled to leave quarantine on the same day, and then they will enter quarantine based on
        # the smallest trace_time, so it should be fine if someone is traced in multiple layers.
        for trace_time, contact_inds in contacts.items():
            sim.people.known_contact[contact_inds] = True
            sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t + trace_time)
            sim.people.schedule_quarantine(contact_inds, start_date=sim.t + trace_time, period=self.quarantine_period - trace_time)  # Schedule quarantine for the notified people to start on the date they will be notified


class AppBasedTracing(DynamicContactTracing):
    def __init__(self, *, coverage: float, layers: list, trace_time: int = 0, **kwargs):
        """
        App based contact tracing parametrized by coverage

        This is similar to normal contact tracing but parametrized by app coverage. The probability
        of tracing a contact is calculated as the probability that both people have the app, so
        is equal to the app coverage squared.

        Arguments must be passed by name to ensure consistency when calling the parent constructor.

        Args:
            coverage: Population app coverage (applies to all layers)
            layers: The layers in which app based tracing can be used
            trace_time: All layers use the same trace time for this intervention
            **kwargs: Optional other arguments including `start_day`, `end_day`, `quarantine_period` - anything supported by `DynamicContactTracing`
        """
        trace_probs = dict.fromkeys(layers, coverage ** 2) # Trace probability is coverage**2 because both people need to have the app
        trace_time = dict.fromkeys(layers, trace_time) # Trace time is the same in all layers
        super().__init__(trace_probs=trace_probs, trace_time=trace_time, **kwargs)


class LimitedContactTracing(DynamicContactTracing):
    """
    Dynamic contact tracing with a capacity limit
    """

    def __init__(self, capacity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity  #: Maximum capacity (number of newly diagnosed people to contact)

    def initialize(self, sim):
        super().initialize(sim)

    def select_cases(self, sim: cv.Sim) -> np.ndarray:
        trace_inds = super().select_cases(sim)

        if self.capacity is None or np.isinf(self.capacity):
            # It would be better if users setting an infinite capacity (or None) would instead
            # use DynamicContactTracing directly, but there is a use case for quickly changing
            # the capacity parameter to disable the capacity limit without changing the
            # intervention type, which we can reasonably accommodate here.
            return trace_inds

        capacity = int(self.capacity / sim.rescale_vec[sim.t])  # Scale capacity based on dynamic rescaling factor (e.g. if the scale factor becomes 2, we need to halve the absolute capacity)
        if len(trace_inds) > capacity:
            trace_inds = np.random.choice(trace_inds, capacity, replace=False)

        return trace_inds