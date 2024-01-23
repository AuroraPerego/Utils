import uproot 
import ROOT
import os
import numpy as np
import pandas as pd
import awkward as ak
import math as m
import mplhep as hep
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from numba import jit, njit, types, prange, typed, typeof, int64, float64
from numba.experimental import jitclass
from numba.typed import List
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

spec_candidate = [
    ('energy', types.float64),
    ('index', types.int64),
    ('charge', types.int64),  # optional for handling empty lists
    ('pdgId', types.int64),  # optional for handling empty lists
    ('track', types.float64),  # optional for handling empty lists
    ('ev', types.int64)
]
@jitclass(spec_candidate)
class TICLCandidate:
   def __init__(self, energy = 0, charge = 0, pdgId = 0, track = -1, index = 0, ev = -1):
       self.energy = energy
       self.index = index
       self.charge = charge
       self.pdgId = pdgId
       self.track = track
       self.ev = ev

@njit
def argmaxNumba(arr):
    if len(arr) == 0:
        raise ValueError("argmax: array is empty")
    
    max_index = 0
    max_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_index = i

    return max_index

@njit
def argminNumba(arr):
    if len(arr) == 0:
        raise ValueError("argmin: array is empty")
    
    min_index = 0
    min_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] < min_value:
            min_value = arr[i]
            min_index = i

    return min_index

@njit
def create_candidate(ticlCandidate_ev, ri, ev):
    energy= ticlCandidate_ev.simTICLCandidate_raw_energy[ri]
    charge = ticlCandidate_ev.simTICLCandidate_charge[ri]
    pdgId = ticlCandidate_ev.simTICLCandidate_pdgId[ri]
    track = ticlCandidate_ev.simTICLCandidate_track_in_candidate[ri]
    
    candidateReco = TICLCandidate(energy, charge, pdgId, track, ri, ev) 
    
    return candidateReco

def create_efficiency_plots(numerator_list, denominator_list, minX, maxX, bins, variable_name, yTitle, title, plotName):
    # Create histograms for passing and total
    h_pass = ROOT.TH1F(f"hist_pass_{variable_name}", f"Passing {variable_name};{variable_name};Counts", bins, minX, maxX)
    h_total = ROOT.TH1F(f"hist_total_{variable_name}", f"Total {variable_name};{variable_name};Counts", bins, minX, maxX)

    # Fill histograms with numerator and denominator values
    for numerator in numerator_list:
        variable_value = getattr(numerator, variable_name)
        h_pass.Fill(variable_value)
        
    for denominator in denominator_list:
        variable_value = getattr(denominator, variable_name)
        h_total.Fill(variable_value)

    # Create a TEfficiency object using histograms
    efficiency = ROOT.TEfficiency(h_pass, h_total)

    # Extract efficiency points and errors
    n_points = efficiency.GetTotalHistogram().GetNbinsX()
    x_values = [efficiency.GetTotalHistogram().GetBinCenter(i) for i in range(1, n_points + 1)]
    y_values = [efficiency.GetEfficiency(i) for i in range(1, n_points + 1)]
    y_errors_low = [efficiency.GetEfficiencyErrorLow(i) for i in range(1, n_points + 1)]
    y_errors_high = [efficiency.GetEfficiencyErrorUp(i) for i in range(1, n_points + 1)]

    # Create Matplotlib plot
    fig = plt.figure(figsize = (15,10))
    plt.errorbar(x_values, y_values, yerr=[y_errors_low, y_errors_high], fmt='o', label='Efficiency')
    plt.xlim(minX, maxX)
    plt.ylim(0, 1)
    plt.xlabel(variable_name)
    plt.ylabel(yTitle)
    plt.title(title)
    plt.legend()

    # Create the "plots" directory if it doesn't exist
    plotDir = 'plotsNumba' 
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    # Save the plot
    plot_filename = os.path.join(plotDir,  f"{plotName}_{variable_name}.png")
    plt.savefig(plot_filename)



fu = "/eos/user/a/aperego/Timing/new_root_files/histov5_noMuCut.root"

files = [fu]

association_data = list() 
TICLCandidates_data = list()
simTICLCandidates_data = list()
tracks_data = list()

for f in tqdm(files):
    file = uproot.open(f)
    associations = file["ticlDumper/associations"]
    tracks = file["ticlDumper/tracks"]
    simTICLCandidates = file["ticlDumper/simTICLCandidate"]
    TICLCandidates = file["ticlDumper/candidates"]

    association_data.append(associations.arrays(associations.keys()))
    tracks_data.append(tracks.arrays(tracks.keys().remove("track_pos_mtd")))
    simTICLCandidates_data.append(simTICLCandidates.arrays(simTICLCandidates.keys()))
    TICLCandidates_data.append(TICLCandidates.arrays(TICLCandidates.keys()))
@njit
def find_track_idx(tracks, idx):
    for i, tr in enumerate(tracks):
        if tr == idx:
            return i
    return -1

#@njit
def process_event(association_data, simTICLCandidates_data, TICLCandidates_data, tracks_data): 
    num_events = 1000
    #candidates_energy = []
    #candidates_pt = []
    #candidates_pdgId = []
    #candidates_type = []
    chg_all_candidates = []
    chg_track_eff_candidates = []
    chg_pid_eff_candidates = []
    chg_energy_eff_candidates = []
    neu_all_candidates = []
    neu_pid_eff_candidates = []
    neu_energy_eff_candidates = []

    for f in prange(len(association_data)):
        association_f = association_data[f]
        simTICLCandidates_f = simTICLCandidates_data[f]
        TICLCandidates_f = TICLCandidates_data[f]
        tracks_f = tracks_data[f]

        for ev in prange(num_events):
            association_ev = association_f[ev]
            simTICLCandidates_ev = simTICLCandidates_f[ev]
            TICLCandidates_ev = TICLCandidates_f[ev]
            tracks_ev = tracks_f[ev]

#            recoToSim_mergeTracksterCP = association_ev.Mergetracksters_recoToSim_CP
#            recoToSim_mergeTracksterCP_score = association_ev.Mergetracksters_recoToSim_CP_score
#            recoToSim_mergeTracksterCP_sharedE = association_ev.Mergetracksters_recoToSim_CP_sharedE
#
            simToReco_mergeTracksterCP = association_ev.Mergetracksters_simToReco_CP
            simToReco_mergeTracksterCP_score = association_ev.Mergetracksters_simToReco_CP_score
            simToReco_mergeTracksterCP_sharedE = association_ev.Mergetracksters_simToReco_CP_sharedE

            # Efficiency and Purity
            for si in range(len(simToReco_mergeTracksterCP)):
                simPid = simTICLCandidates_ev.simTICLCandidate_pdgId[si]
                simEnergy = simTICLCandidates_ev.simTICLCandidate_raw_energy[si]
                #argmaxShared = argmaxNumba(simToReco_mergeTracksterCP_sharedE[si])
                argminScore= argminNumba(simToReco_mergeTracksterCP_score[si])
                minScore = simToReco_mergeTracksterCP_score[si][argminScore]
                maxSE = 0.
                candIdx = -1
                if minScore < 1.:
                    maxSE = simToReco_mergeTracksterCP_sharedE[si][argminScore]
                    candIdx = simToReco_mergeTracksterCP[si][argminScore]
                if candIdx == -1:
                    continue
                recoPid = TICLCandidates_ev.candidate_pdgId[candIdx]

                if simTICLCandidates_ev.simTICLCandidate_charge[si]:
                    simTrk = simTICLCandidates_ev.simTICLCandidate_track_in_candidate[si]
                    if simTrk == -1:
                        continue
                    id_in_trkColl = find_track_idx(tracks_ev.track_id, simTrk) #np.where(tracks_ev.track_id == simTrk)[0][0]
                    if id_in_trkColl == -1:
                        continue
                    # TODO: add also muId cut and <2 GeV cut??
                    if tracks_ev.track_pt[id_in_trkColl] < 1 or tracks_ev.track_missing_outer_hits[id_in_trkColl] > 5 or not tracks_ev.track_quality[id_in_trkColl]: # GeV
                        continue
                    simTICLCandidate = create_candidate(simTICLCandidates_ev, si, ev)
                    chg_all_candidates.append(simTICLCandidate)

                    recoTrk = TICLCandidates_ev.track_in_candidate[candIdx]
                    if simTrk == recoTrk:
                        chg_track_eff_candidates.append(simTICLCandidate)
                        if simPid == recoPid:
                            chg_pid_eff_candidates.append(simTICLCandidate)
                            if maxSE / simEnergy > 0.5:
                                chg_energy_eff_candidates.append(simTICLCandidate)
                else:
                    simTICLCandidate = create_candidate(simTICLCandidates_ev, si, ev)
                    neu_all_candidates.append(simTICLCandidate)

                    if simPid == recoPid:
                        neu_pid_eff_candidates.append(simTICLCandidate)
                        if maxSE / simEnergy > 0.5:
                            neu_energy_eff_candidates.append(simTICLCandidate)
                        
    return chg_all_candidates, chg_track_eff_candidates, chg_pid_eff_candidates, chg_energy_eff_candidates, neu_all_candidates, neu_pid_eff_candidates, neu_energy_eff_candidates

chg_all_candidates, chg_track_eff_candidates, chg_pid_eff_candidates, chg_energy_eff_candidates, neu_all_candidates, neu_pid_eff_candidates, neu_energy_eff_candidates = process_event(association_data, simTICLCandidates_data, TICLCandidates_data, tracks_data)

#create_efficiency_plots(chg_track_eff_candidates, chg_all_candidates, 1.5 , 3.0, 10, "eta", "Track efficiency", "Track efficiency - charged", 'effTrk')
#create_efficiency_plots(chg_track_eff_candidates, chg_all_candidates, -m.pi , m.pi, 10, "phi", "Track efficiency", "Track efficiency - charged", 'effTrk')
create_efficiency_plots(chg_track_eff_candidates, chg_all_candidates, 0 , 3000, 20, "energy", "Track efficiency", "Track efficiency - charged", 'effTrk_charged')

#create_efficiency_plots(chg_pid_eff_candidates, chg_all_candidates, 1.5 , 3.0, 10, "eta", "PdgID Efficiency", "PdgID Efficiency - charged", 'effPid')
#create_efficiency_plots(chg_pid_eff_candidates, chg_all_candidates, -m.pi , m.pi, 10, "phi", "PdgID Efficiency", "PdgID Efficiency - charged", 'effPid')
create_efficiency_plots(chg_pid_eff_candidates, chg_all_candidates, 0 , 3000, 20, "energy", "PdgID Efficiency", "PdgID Efficiency - charged", 'effPid_charged')

#create_efficiency_plots(chg_energy_eff_candidates, chg_all_candidates, 1.5 , 3.0, 10, "eta", "Energy Efficiency", "Energy Efficiency - charged", 'effEnergy')
#create_efficiency_plots(chg_energy_eff_candidates, chg_all_candidates, -m.pi , m.pi, 10, "phi", "Energy Efficiency", "Energy Efficiency - charged", 'effEnergy')
create_efficiency_plots(chg_energy_eff_candidates, chg_all_candidates, 0 , 3000, 20, "energy", "Energy Efficiency", "Energy Efficiency - charged", 'effEnergy_charged')

#create_efficiency_plots(neu_pid_eff_candidates, neu_all_candidates, 1.5 , 3.0, 10, "eta", "PdgID Efficiency", "PdgID Efficiency - neutral", 'effPid')
#create_efficiency_plots(neu_pid_eff_candidates, neu_all_candidates, -m.pi , m.pi, 10, "phi", "PdgID Efficiency", "PdgID Efficiency - neutral", 'effPid')
create_efficiency_plots(neu_pid_eff_candidates, neu_all_candidates, 0 , 3000, 20, "energy", "PdgID Efficiency", "PdgID Efficiency - neutral", 'effPid_neutral')

#create_efficiency_plots(neu_energy_eff_candidates, neu_all_candidates, 1.5 , 3.0, 10, "eta", "Energy Efficiency", "Energy Efficiency - neutral", 'effEnergy')
#create_efficiency_plots(neu_energy_eff_candidates, neu_all_candidates, -m.pi , m.pi, 10, "phi", "Energy Efficiency", "Energy Efficiency - neutral", 'effEnergy')
create_efficiency_plots(neu_energy_eff_candidates, neu_all_candidates, 0 , 3000, 20, "energy", "Energy Efficiency", "Energy Efficiency - neutral", 'effEnergy_neutral')
