import uproot 
import ROOT
import os
import numpy as np
import pandas as pd
import awkward as ak
import math as m
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib as mpl
from enum import Enum
from numba import jit, njit, types, prange, typed, typeof, int64, float64
from numba.experimental import jitclass
from numba.typed import List
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
mpl.rc('legend', fontsize=16) 
mpl.rc('axes', labelsize=18, titlesize=20)
plt.style.use(hep.style.CMS)

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
def create_simcandidate(ticlCandidate_ev, ri, ev):
    energy= ticlCandidate_ev.simTICLCandidate_raw_energy[ri]
    charge = ticlCandidate_ev.simTICLCandidate_charge[ri]
    pdgId = ticlCandidate_ev.simTICLCandidate_pdgId[ri]
    track = ticlCandidate_ev.simTICLCandidate_track_in_candidate[ri]
    
    candidateSim = TICLCandidate(energy, charge, pdgId, track, ri, ev) 
    
    return candidateSim

@njit
def create_recocandidate(ticlCandidate_ev, ri, ev):
    energy= ticlCandidate_ev.candidate_raw_energy[ri]
    charge = ticlCandidate_ev.candidate_charge[ri]
    pdgId = ticlCandidate_ev.candidate_pdgId[ri]
    track = ticlCandidate_ev.track_in_candidate[ri]
    
    candidateReco = TICLCandidate(energy, charge, pdgId, track, ri, ev) 
    
    return candidateReco

class ParticleType(Enum):
    neutral_pion = 1
    photon = 2
    electron = 3
    muon = 4
    charged_hadron = 5
    neutral_hadron = 6
    unknown = 7

@njit
def particle_type_from_pdg_id(pdg_id, charge):
    if pdg_id == 111:
        return ParticleType.neutral_pion
    else:
        pdg_id = abs(pdg_id)
        if pdg_id == 22:
            return ParticleType.photon
        elif pdg_id == 11:
            return ParticleType.electron
        elif pdg_id == 13:
            return ParticleType.muon
        else:
            is_hadron = (pdg_id > 100 and pdg_id < 900) or (pdg_id > 1000 and pdg_id < 9000)
            if is_hadron:
                if charge != 0:
                    return ParticleType.charged_hadron
                else:
                    return ParticleType.neutral_hadron
            else:
                return ParticleType.unknown

def create_efficiency_plots(numerator_list, denominator_list, minX, maxX, bins, variable_name, xTitle, yTitle, title, plotName, fileName):
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
    plt.errorbar(x_values, y_values, yerr=[y_errors_low, y_errors_high], fmt='o', label='Efficiency', capsize=2)
    plt.xlim(minX, maxX)
    plt.ylim(0, 1)
    plt.xlabel(xTitle, fontsize=20)
    plt.ylabel(yTitle, fontsize=20)
    plt.title(title, fontsize=22)
    #plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)

    # Create the "plots" directory if it doesn't exist
    plotDir = 'plotsNumba' 
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    # Save the plot
    plot_filename = os.path.join(plotDir,  f"{fileName}_{plotName}_{variable_name}.png")
    plt.savefig(plot_filename)

def create_stack_efficiency_plots(numerator_list, denominator_list, minX, maxX, bins, variable_name, xTitle, yTitle, mainTitle, plotName, fileName):
    # Create histogram for denominator values
    h_total = ROOT.TH1F(f"hist_total_{variable_name}", f"Total {variable_name};{variable_name};Counts", bins, minX, maxX)
    # Fill histogram with denominator values
    for denominator in denominator_list:
        variable_value = getattr(denominator, variable_name)
        h_total.Fill(variable_value)

    # Create Matplotlib plot
    fig = plt.figure(figsize = (15,10))
    # loop over the numerators
    for numerators, title in zip(numerator_list, yTitle):
        # Create histograms for passing and total
        h_pass = ROOT.TH1F(f"hist_pass_{variable_name}", f"Passing {variable_name};{variable_name};Counts", bins, minX, maxX)

        # Fill histograms with numerator and denominator values
        for numerator in numerators:
            variable_value = getattr(numerator, variable_name)
            h_pass.Fill(variable_value)

        # Create a TEfficiency object using histograms
        efficiency = ROOT.TEfficiency(h_pass, h_total)

        # Extract efficiency points and errors
        n_points = efficiency.GetTotalHistogram().GetNbinsX()
        x_values = [efficiency.GetTotalHistogram().GetBinCenter(i) for i in range(1, n_points + 1)]
        y_values = [efficiency.GetEfficiency(i) for i in range(1, n_points + 1)]
        y_errors_low = [efficiency.GetEfficiencyErrorLow(i) for i in range(1, n_points + 1)]
        y_errors_high = [efficiency.GetEfficiencyErrorUp(i) for i in range(1, n_points + 1)]

        plt.errorbar(x_values, y_values, yerr=[y_errors_low, y_errors_high], fmt='o', label=title, capsize=2)
        del h_pass

    plt.xlim(minX, maxX)
    plt.ylim(0, 1)
    plt.xlabel(xTitle, fontsize=20)
    plt.ylabel("Efficiency", fontsize=20)
    plt.title(mainTitle, fontsize=22)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xticks(fontsize=18)

    # Create the "plots" directory if it doesn't exist
    plotDir = 'plotsNumba' 
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    # Save the plot
    plot_filename = os.path.join(plotDir,  f"{fileName}_{plotName}_{variable_name}.png")
    plt.savefig(plot_filename)

@njit
def find_track_idx(tracks, idx):
    for i, tr in enumerate(tracks):
        if tr == idx:
            return i
    return -1

@njit(parallel=True)
def process_event(association_data, simTICLCandidates_data, TICLCandidates_data, tracks_data): 
    num_events = 500

    chg_all_sim_candidates = []
    chg_track_eff_sim_candidates = []
    chg_pid_eff_sim_candidates = []
    chg_energy_eff_sim_candidates = []
    neu_all_sim_candidates = []
    neu_pid_eff_sim_candidates = []
    neu_energy_eff_sim_candidates = []

    chg_all_candidates = []
    chg_track_fake_candidates = []
    chg_pid_fake_candidates = []
    chg_energy_fake_candidates = []
    neu_all_candidates = []
    neu_pid_fake_candidates = []
    neu_energy_fake_candidates = []

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

            simToReco_mergeTracksterCP = association_ev.Mergetracksters_simToReco_CP
            simToReco_mergeTracksterCP_score = association_ev.Mergetracksters_simToReco_CP_score
            simToReco_mergeTracksterCP_sharedE = association_ev.Mergetracksters_simToReco_CP_sharedE

            # Efficiency
            for si in range(len(simToReco_mergeTracksterCP)):
                simPid = simTICLCandidates_ev.simTICLCandidate_pdgId[si]
                simCharge = simTICLCandidates_ev.simTICLCandidate_charge[si]
                simEnergy = simTICLCandidates_ev.simTICLCandidate_raw_energy[si]
                #argmaxShared = argmaxNumba(simToReco_mergeTracksterCP_sharedE[si])
                argminScore= argminNumba(simToReco_mergeTracksterCP_score[si])
                minScore = simToReco_mergeTracksterCP_score[si][argminScore]
                maxSE = 0.
                candIdx = -1
                if minScore < 1.:
                    maxSE = simToReco_mergeTracksterCP_sharedE[si][argminScore]
                    for i, k in enumerate(TICLCandidates_ev.tracksters_in_candidate):
                        if not len(k):
                            continue
                        for kk in k:
                            if kk == simToReco_mergeTracksterCP[si][argminScore]:
                                candIdx = i
                                break
                if candIdx == -1:
                    continue
                recoPid = TICLCandidates_ev.candidate_pdgId[candIdx]
                recoCharge = TICLCandidates_ev.candidate_charge[candIdx]

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
                    simTICLCandidate = create_simcandidate(simTICLCandidates_ev, si, ev)
                    chg_all_sim_candidates.append(simTICLCandidate)

                    recoTrk = TICLCandidates_ev.track_in_candidate[candIdx]
                    if simTrk == recoTrk:
                        chg_track_eff_sim_candidates.append(simTICLCandidate)
                        if particle_type_from_pdg_id(simPid, simCharge) == particle_type_from_pdg_id(recoPid, recoCharge):
                            chg_pid_eff_sim_candidates.append(simTICLCandidate)
                            if maxSE / simEnergy > 0.5:
                                chg_energy_eff_sim_candidates.append(simTICLCandidate)
                else:
                    simTICLCandidate = create_simcandidate(simTICLCandidates_ev, si, ev)
                    neu_all_sim_candidates.append(simTICLCandidate)

                    if particle_type_from_pdg_id(simPid, simCharge) == particle_type_from_pdg_id(recoPid, recoCharge):
                        neu_pid_eff_sim_candidates.append(simTICLCandidate)
                        if maxSE / simEnergy > 0.5:
                            neu_energy_eff_sim_candidates.append(simTICLCandidate)
                recoEnergy = TICLCandidates_ev.candidate_raw_energy[candIdx]

            recoToSim_mergeTracksterCP = association_ev.Mergetracksters_recoToSim_CP
            recoToSim_mergeTracksterCP_score = association_ev.Mergetracksters_recoToSim_CP_score
            recoToSim_mergeTracksterCP_sharedE = association_ev.Mergetracksters_recoToSim_CP_sharedE

            # Fake
            for ri in range(len(recoToSim_mergeTracksterCP)):
                recoPid = TICLCandidates_ev.candidate_pdgId[ri]
                recoCharge = TICLCandidates_ev.candidate_charge[ri]
                recoEnergy = TICLCandidates_ev.candidate_raw_energy[ri]
                ts_idx = TICLCandidates_ev.tracksters_in_candidate[ri]
                if len(ts_idx) == 0:
                    continue
                ts_idx = ts_idx[0]
                argminScore= argminNumba(recoToSim_mergeTracksterCP_score[ts_idx])
                minScore = recoToSim_mergeTracksterCP_score[ts_idx][argminScore]
                maxSE = 0.
                candIdx = -1
                si = -1
                if minScore < 1.:
                    maxSE = recoToSim_mergeTracksterCP_sharedE[ts_idx][argminScore]
                    si = recoToSim_mergeTracksterCP[ts_idx][argminScore]
                if si == -1:
                    continue
                simPid = simTICLCandidates_ev.simTICLCandidate_pdgId[si]
                simCharge = simTICLCandidates_ev.simTICLCandidate_charge[si]

                if recoCharge:
                    simTrk = simTICLCandidates_ev.simTICLCandidate_track_in_candidate[si]
                    if simTrk == -1 and (np.abs(simPid) == 211 or np.abs(simPid) == 11):
                        # charged without a reco track associated with it: skip
                        continue
                    TICLCandidate = create_recocandidate(TICLCandidates_ev, ri, ev)
                    chg_all_candidates.append(TICLCandidate)
                    recoTrk = TICLCandidates_ev.track_in_candidate[ri]
                    if simTrk != recoTrk: # linked to the wrong track or sim is neutral
                        chg_track_fake_candidates.append(TICLCandidate)
                    if particle_type_from_pdg_id(simPid, simCharge) != particle_type_from_pdg_id(recoPid, recoCharge):
                        chg_pid_fake_candidates.append(TICLCandidate)
                    if maxSE / recoEnergy < 0.5:
                        chg_energy_fake_candidates.append(TICLCandidate)
                else:
                    TICLCandidate = create_recocandidate(TICLCandidates_ev, ri, ev)
                    neu_all_candidates.append(TICLCandidate)

                    if particle_type_from_pdg_id(simPid, simCharge) != particle_type_from_pdg_id(recoPid, recoCharge):
                        neu_pid_fake_candidates.append(TICLCandidate)
                    if maxSE / recoEnergy < 0.5:
                        neu_energy_fake_candidates.append(TICLCandidate)


    return chg_all_sim_candidates, chg_track_eff_sim_candidates, chg_pid_eff_sim_candidates, chg_energy_eff_sim_candidates, neu_all_sim_candidates, neu_pid_eff_sim_candidates, neu_energy_eff_sim_candidates, chg_all_candidates, chg_track_fake_candidates, chg_pid_fake_candidates, chg_energy_fake_candidates, neu_all_candidates, neu_pid_fake_candidates, neu_energy_fake_candidates

##########
#        #
#  MAIN  #
#        #
##########

fu = "/eos/user/a/aperego/Timing/new_root_files/D99/histo_0PU_v5.root"
#fu = "/eos/user/a/aperego/SampleProduction/ParticleGunPionPU200/histo_200PU_v5.root"
#fu = "/eos/user/a/aperego/SampleProduction/ParticleGunPionPU200_v5/histo_200PU_v5_new.root"
#fu = "/eos/user/a/aperego/SampleProduction/ParticleGunPionPU200_v4/histo_200PU_v4.root"
#fu = "/data2/user/aperego/cmssw/PRtimingFelice/25288.0_SinglePiPt25Eta1p7_2p7+2026D99/histo.root"
fileName = fu.split("/")[-1].replace(".root","")

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
    try:
        tracks_data.append(tracks.arrays(tracks.keys().remove("track_pos_mtd")))
    except:
        tracks_data.append(tracks.arrays(tracks.keys()))
    simTICLCandidates_data.append(simTICLCandidates.arrays(simTICLCandidates.keys()))
    TICLCandidates_data.append(TICLCandidates.arrays(TICLCandidates.keys()))

chg_all_sim_candidates, chg_track_eff_sim_candidates, chg_pid_eff_sim_candidates, chg_energy_eff_sim_candidates, neu_all_sim_candidates, neu_pid_eff_sim_candidates, neu_energy_eff_sim_candidates, chg_all_candidates, chg_track_fake_candidates, chg_pid_fake_candidates, chg_energy_fake_candidates, neu_all_candidates, neu_pid_fake_candidates, neu_energy_fake_candidates = process_event(association_data, simTICLCandidates_data, TICLCandidates_data, tracks_data)

#create_efficiency_plots(chg_track_eff_sim_candidates, chg_all_sim_candidates, 1.5 , 3.0, 10, "eta", "Track efficiency", "Track efficiency - charged", 'effTrk', fileName)
#create_efficiency_plots(chg_track_eff_sim_candidates, chg_all_sim_candidates, -m.pi , m.pi, 10, "phi", "Track efficiency", "Track efficiency - charged", 'effTrk', fileName)
create_efficiency_plots(chg_track_eff_sim_candidates, chg_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", "Track efficiency", "Efficiency of track linking - charged candidates", 'effTrk_charged', fileName)

#create_efficiency_plots(chg_pid_eff_sim_candidates, chg_all_sim_candidates, 1.5 , 3.0, 10, "eta", "PID Efficiency", "PdgID Efficiency - charged", 'effPid', fileName)
#create_efficiency_plots(chg_pid_eff_sim_candidates, chg_all_sim_candidates, -m.pi , m.pi, 10, "phi", "PID Efficiency", "PdgID Efficiency - charged", 'effPid', fileName)
create_efficiency_plots(chg_pid_eff_sim_candidates, chg_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", "PID Efficiency", "Efficiency of PID assignment (EM / HAD) - charged candidates", 'effPid_charged', fileName)

#create_efficiency_plots(chg_energy_eff_sim_candidates, chg_all_sim_candidates, 1.5 , 3.0, 10, "eta", "Energy Efficiency", "Energy Efficiency - charged", 'effEnergy', fileName)
#create_efficiency_plots(chg_energy_eff_sim_candidates, chg_all_sim_candidates, -m.pi , m.pi, 10, "phi", "Energy Efficiency", "Energy Efficiency - charged", 'effEnergy', fileName)
create_efficiency_plots(chg_energy_eff_sim_candidates, chg_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", "Energy Efficiency", "Efficiency of energy reconstruction (shared energy > 50%) - charged candidates", 'effEnergy_charged', fileName)

#create_efficiency_plots(neu_pid_eff_sim_candidates, neu_all_sim_candidates, 1.5 , 3.0, 10, "eta", "PID Efficiency", "PdgID Efficiency - neutral", 'effPid', fileName)
#create_efficiency_plots(neu_pid_eff_sim_candidates, neu_all_sim_candidates, -m.pi , m.pi, 10, "phi", "PID Efficiency", "PdgID Efficiency - neutral", 'effPid', fileName)
create_efficiency_plots(neu_pid_eff_sim_candidates, neu_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", "PID Efficiency", "Efficiency of PID assignment (EM / HAD) - neutral candidates", 'effPid_neutral', fileName)

#create_efficiency_plots(neu_energy_eff_sim_candidates, neu_all_sim_candidates, 1.5 , 3.0, 10, "eta", "Energy Efficiency", "Energy Efficiency - neutral", 'effEnergy', fileName)
#create_efficiency_plots(neu_energy_eff_sim_candidates, neu_all_sim_candidates, -m.pi , m.pi, 10, "phi", "Energy Efficiency", "Energy Efficiency - neutral", 'effEnergy', fileName)
create_efficiency_plots(neu_energy_eff_sim_candidates, neu_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", "Energy Efficiency", "Efficiency of energy reconstruction (shared energy > 50%) - neutral candidates", 'effEnergy_neutral', fileName)

create_stack_efficiency_plots([chg_track_eff_sim_candidates, chg_pid_eff_sim_candidates, chg_energy_eff_sim_candidates], chg_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", ["Track efficiency", "PID Efficiency", "Energy Efficiency"], "Stacked efficiencies for charged candidates", 'effStacked_charged', fileName)
create_stack_efficiency_plots([neu_pid_eff_sim_candidates, neu_energy_eff_sim_candidates], neu_all_sim_candidates, 0 , 1200, 10, "energy", "simTICLCandidate Energy (GeV)", ["PID Efficiency", "Energy Efficiency"], "Stacked efficiencies for neutral candidates", 'effStacked_neutral', fileName)

#create_efficiency_plots(chg_track_fake_candidates, chg_all_candidates, 1.5 , 3.0, 10, "eta", "Track fake", "Track fake - charged", 'fakeTrk', fileName)
#create_efficiency_plots(chg_track_fake_candidates, chg_all_candidates, -m.pi , m.pi, 10, "phi", "Track fake", "Track fake - charged", 'fakeTrk', fileName)
create_efficiency_plots(chg_track_fake_candidates, chg_all_candidates, 0 , 1200, 10, "energy", "TICLCandidate Energy (GeV)", "Track fake", "Wrong (or not linked) tracks - charged candidates", 'fakeTrk_charged', fileName)

#create_efficiency_plots(chg_pid_fake_candidates, chg_all_candidates, 1.5 , 3.0, 10, "eta", "PID fake", "PdgID fake - charged", 'fakePid', fileName)
#create_efficiency_plots(chg_pid_fake_candidates, chg_all_candidates, -m.pi , m.pi, 10, "phi", "PID fake", "PdgID fake - charged", 'fakePid', fileName)
create_efficiency_plots(chg_pid_fake_candidates, chg_all_candidates, 0 , 1200, 10, "energy", "TICLCandidate Energy (GeV)", "PID fake", "Wrong PID assignment - charged candidates", 'fakePid_charged', fileName)

#create_efficiency_plots(chg_energy_fake_candidates, chg_all_candidates, 1.5 , 3.0, 10, "eta", "Energy fake", "Energy fake - charged", 'fakeEnergy', fileName)
#create_efficiency_plots(chg_energy_fake_candidates, chg_all_candidates, -m.pi , m.pi, 10, "phi", "Energy fake", "Energy fake - charged", 'fakeEnergy', fileName)
create_efficiency_plots(chg_energy_fake_candidates, chg_all_candidates, 0 , 1200, 10, "energy", "TICLCandidate Energy (GeV)", "Energy fake", "Less than 50% of energy reconstructed - charged candidates", 'fakeEnergy_charged', fileName)

#create_efficiency_plots(neu_pid_fake_candidates, neu_all_candidates, 1.5 , 3.0, 10, "eta", "PID fake", "PdgID fake - neutral", 'fakePid', fileName)
#create_efficiency_plots(neu_pid_fake_candidates, neu_all_candidates, -m.pi , m.pi, 10, "phi", "PID fake", "PdgID fake - neutral", 'fakePid', fileName)
create_efficiency_plots(neu_pid_fake_candidates, neu_all_candidates, 0 , 1200, 10, "energy", "TICLCandidate Energy (GeV)", "PID fake", "Wrong PID assignment - neutral candidates", 'fakePid_neutral', fileName)

#create_efficiency_plots(neu_energy_fake_candidates, neu_all_candidates, 1.5 , 3.0, 10, "eta", "Energy fake", "Energy fake - neutral", 'fakeEnergy', fileName)
#create_efficiency_plots(neu_energy_fake_candidates, neu_all_candidates, -m.pi , m.pi, 10, "phi", "Energy fake", "Energy fake - neutral", 'fakeEnergy', fileName)
create_efficiency_plots(neu_energy_fake_candidates, neu_all_candidates, 0 , 1200, 10, "energy", "TICLCandidate Energy (GeV)", "Energy fake", "Less than 50% of energy reconstructed - neutral candidates", 'fakeEnergy_neutral', fileName)
