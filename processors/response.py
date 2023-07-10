import numpy as np
import awkward as ak
from distributed import Client
import matplotlib.pyplot as plt
import mplhep
import pandas as pd
import coffea.util
import re
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema
from coffea import processor
import hist
from hist import Hist
from coffea.analysis_tools import PackedSelection
import random
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from collections import defaultdict

from processors.helper import (
    run_deltar_matching,
    require_n,
    criteria_one,
    compute_asymmetry,
    select_eta,
    add_pileup_weight,
)

class ResponseProcessor(processor.ProcessorABC):
    def __init__(self):
        
        # scouting JEC
        ext = extractor()
        ext.add_weight_sets([
            "* * data/jec/Run3Summer21_V2_MC_L1FastJet_AK4PFchsHLT.txt",
            "* * data/jec/Run3Summer21_V2_MC_L2Relative_AK4PFchsHLT.txt",
            "* * data/jec/Run3Summer21_V2_MC_L3Absolute_AK4PFchsHLT.txt",
        ])
        ext.finalize()
        evaluator = ext.make_evaluator()

        jec_stack_names = evaluator.keys()
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)
        
        name_map = jec_stack.blank_name_map
        name_map['JetPt'] = 'pt'
        name_map['JetMass'] = 'mass'
        name_map['JetEta'] = 'eta'
        name_map['JetA'] = 'area'
        name_map['ptRaw'] = 'pt_raw'
        name_map['massRaw'] = 'mass_raw'
        name_map['Rho'] = 'rho'
        
        self._jet_factory_scouting = CorrectedJetsFactory(name_map, jec_stack)
        
        # offline JEC
        ext = extractor()
#         ext.add_weight_sets([
#             "* * data/jec/Winter22Run3_V1_MC_L1FastJet_AK4PFchs.txt",
#             "* * data/jec/Winter22Run3_V1_MC_L2Relative_AK4PFchs.txt",
#             "* * data/jec/Winter22Run3_V1_MC_L3Absolute_AK4PFchs.txt",
#         ])
        ext.add_weight_sets([
           "* * data/jec/Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi.txt",
           "* * data/jec/Winter22Run3_V2_MC_L2Relative_AK4PFPuppi.txt",
           "* * data/jec/Winter22Run3_V2_MC_L3Absolute_AK4PFPuppi.txt",
        ])
#         ext.add_weight_sets([
#            "* * data/jec/Summer22EEPrompt22_V1_MC_L1FastJet_AK4PFPuppi.txt",
#            "* * data/jec/Summer22EEPrompt22_V1_MC_L2Relative_AK4PFPuppi.txt",
#            "* * data/jec/Summer22EEPrompt22_V1_MC_L3Absolute_AK4PFPuppi.txt",
#         ])
        ext.finalize()
        evaluator = ext.make_evaluator()

        jec_stack_names = evaluator.keys()
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)
        
        name_map = jec_stack.blank_name_map
        name_map['JetPt'] = 'pt'
        name_map['JetMass'] = 'mass'
        name_map['JetEta'] = 'eta'
        name_map['JetA'] = 'area'
        name_map['ptRaw'] = 'pt_raw'
        name_map['massRaw'] = 'mass_raw'
        name_map['Rho'] = 'rho'
        
        self._jet_factory_offline = CorrectedJetsFactory(name_map, jec_stack)

    def process(self, events):
        
        events = events[
            (events.Pileup.nPU < 100)
        ]
        
        dataset = events.metadata['dataset']
        output = defaultdict()
        output["nevents"] = len(events)
        add_pileup_weight(events)
        
        h = {
            "scouting" : Hist(
                hist.axis.Variable([0, 85, 115, 145, 165, 210, 230, 295, 360, 445, 495, 550, 600] + list(np.arange(650, 2050, 50)), name="pt_ave", label=r"$p_{T}^{ave}$"),
                hist.axis.StrCategory([], name="eta", label=r"|$\eta$|", growth=True),
                hist.axis.Regular(100, 0.5, 1.5, name="ratio", label=r"Leading reco $p_T$/Leading gen $p_T$"),
                hist.axis.StrCategory([], name="match", label=r"Matched", growth=True),
                hist.axis.Variable([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], name="alpha", label=r"$\alpha$"),
                hist.axis.StrCategory([], name="dijet", label=r"Dijet", growth=True)
            ),
            "offline" : Hist(
                hist.axis.Variable([0, 85, 115, 145, 165, 210, 230, 295, 360, 445, 495, 550, 600] + list(np.arange(650, 2050, 50)), name="pt_ave", label=r"$p_{T}^{ave}$"),
                hist.axis.StrCategory([], name="eta", label=r"|$\eta$|", growth=True),
                hist.axis.Regular(100, 0.5, 1.5, name="ratio", label=r"Leading reco $p_T$/Leading gen $p_T$"),
                hist.axis.StrCategory([], name="match", label=r"Matched", growth=True),
                hist.axis.Variable([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], name="alpha", label=r"$\alpha$"),
                hist.axis.StrCategory([], name="dijet", label=r"Dijet", growth=True)
            ),
        }

        def apply_jec(jets, rho_name, events):

            if "Scouting" in rho_name:

                jets["pt_raw"] = jets["pt"]
                jets["mass_raw"] = jets["mass"]
                jets['rho'] = ak.broadcast_arrays(events[rho_name], jets.pt)[0]
                corrected_jets = self._jet_factory_scouting.build(jets, lazy_cache=events.caches[0])

            else:

                jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
                jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
                jets['rho'] = ak.broadcast_arrays(events.Rho[rho_name], jets.pt)[0]

                corrected_jets = self._jet_factory_offline.build(jets, lazy_cache=events.caches[0])

            return corrected_jets

        # select good scouting muons
        muons_s = events.ScoutingMuon
        muons_s = muons_s[
            (muons_s.pt > 10)
            & (abs(muons_s.eta) < 2.4)
            & (abs(muons_s.trk_dxy) < 0.2)
            & (abs(muons_s.trackIso) < 0.15)
            & (abs(muons_s.trk_dz) < 0.5)
            #& (muons_s["type"] == 2)
            & (muons_s.normchi2 < 10)
            & (muons_s.nValidRecoMuonHits > 0)
            & (muons_s.nRecoMuonMatchedStations > 1)
            & (muons_s.nValidPixelHits > 0)
            & (muons_s.nTrackerLayersWithMeasurement > 5)

        ]

        # select good offline muons
        muons_o = events.Muon
        muons_o = muons_o[
            (muons_o.pt > 10)
            & (abs(muons_o.eta) < 2.4)
            & (muons_o.pfRelIso04_all < 0.25)
            & (muons_o.looseId)
        ]

        # select good scouting jets
        jets_s = events.ScoutingJet
        jets_s['weight_pileup'] = ak.broadcast_arrays(events['weight_pileup'], jets_s.pt)[0]
        jets_s = apply_jec(jets_s, "ScoutingRho", events)
        jets_s = jets_s[
            (abs(jets_s.eta) < 2.5)
            & (jets_s.pt > 12)
            & (jets_s.neHEF < 0.9)
            & (jets_s.neEmEF < 0.9)
            & (jets_s.muEmEF < 0.8)
            & (jets_s.chHEF > 0.01)
            & (jets_s.chEmEF < 0.8)
            & ak.all(jets_s.metric_table(muons_s) > 0.4, axis=-1)
        ]

        # select good offline jets
        jets_o = events.Jet
        jets_o['weight_pileup'] = ak.broadcast_arrays(events['weight_pileup'], jets_o.pt)[0]
        jets_o = apply_jec(jets_o, "fixedGridRhoFastjetAll", events)
        jets_o = jets_o[
            (abs(jets_o.eta) < 2.5)
            & (jets_o.pt > 12)
            & (jets_o.neHEF < 0.9)
            & (jets_o.neEmEF < 0.9)
            & (jets_o.muEF < 0.8)
            & (jets_o.chHEF > 0.01)
            & (jets_o.chEmEF < 0.8)
            & ak.all(jets_o.metric_table(muons_o) > 0.4, axis=-1)
        ]
        
        jets_gen = events.GenJet
        jets_gen['weight_pileup'] = ak.broadcast_arrays(events['weight_pileup'], jets_gen.pt)[0]

        # require at least one jet in each event
        jet_s = jets_s[
            (ak.num(jets_s) > 1)
        ]
        jet_o = jets_o[
            (ak.num(jets_o) > 1)
        ]

        jet_gen = jets_gen[
            (ak.num(jets_gen) > 1)
        ]
        
        gens_orig = jets_gen
        for eta_region in ["barrel", "endcap"]:
            for rec in ["offline", "scouting"]:

                if (rec == "scouting"):
                    jets_orig = jets_s
                    pt_type = "pt_jec"
                elif (rec == "offline"):
                    jets_orig = jets_o
                    pt_type = "pt_jec"

                jets_cut = jets_orig[
                    (ak.num(jets_orig) > 1)
                    & (ak.num(gens_orig) > 1)
                ]
                gens_cut = gens_orig[
                    (ak.num(jets_orig) > 1)
                    & (ak.num(gens_orig) > 1)
                ]

                # select eta region
                eta_cut = select_eta(eta_region, jets_cut, gens_cut)
                jets_cut = jets_cut[eta_cut]
                gens_cut = gens_cut[eta_cut]

                # loop over different angular matching requirements
                for dr in [0, 0.2, 0.1, 0.005]:
                    
                    if dr == 0:
                    # no matching between reco and gen jet required
                        jets_match = jets_cut
                        gens_match = gens_cut
                    else:
                    # match reco and gen jet
                        match = (
                            ak.firsts(jets_cut).delta_r(ak.firsts(gens_cut)) < dr
                        )

                        jets_match = jets_cut[match]
                        gens_match = gens_cut[match]

                    # fill histogram
                    h[f"{rec}"].fill(
                        pt_ave = gens_match[:,0].pt,
                        eta = eta_region,
                        ratio = jets_match[:,0][pt_type] / gens_match[:,0].pt,
                        match=f'{"No" if dr == 0 else dr}',
                        alpha = 0.0,
                        dijet="False",
                        weight = jets_match[:,0]['weight_pileup'],
                    )
                    
                    # require dijet event
                    # loop over the requirement that the event has exactly two jets or exactly more than two
                    for two in [True, False]:

                        jets_match2, gens_match2 = require_n(jets_match, gens_match, two=two)
                        # dijet criteria
                        jets_match2, gens_match2 = criteria_one(jets_match2, gens_match2, phi=2.7)
                        
                        h[f"{rec}"].fill(
                            pt_ave = gens_match2[:,0].pt,
                            eta = eta_region,
                            ratio = jets_match2[:,0][pt_type] / gens_match2[:,0].pt,
                            match=f'{"No" if dr == 0 else dr}',
                            alpha = 0.0 if two else (jets_match2[:,2][pt_type] / ((jets_match2[:,0][pt_type] + jets_match2[:,1][pt_type]) / 2)),
                            dijet="True",
                            weight = jets_match2[:,0]['weight_pileup'],
                        )

            output[dataset] = h 
        
        return output

    def postprocess(self, accumulator):
        pass
