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

class TriggerProcessor(processor.ProcessorABC):
    def __init__(self):
        
        self._triggers = {
            'L1_SingleJet60' : 60,
            'L1_SingleJet90' : 90,
            'L1_SingleJet120' : 120,
            'L1_SingleJet180' : 180,
            'L1_SingleJet200' : 200,
            'HLT_PFJet60' : 60,
            'HLT_PFJet80' : 80,
            'HLT_PFJet140' : 140,
            'HLT_PFJet200' : 200,
            'HLT_PFJet260' : 260,
            'HLT_PFJet320' : 320,
            'HLT_PFJet400' : 400, 
            'HLT_PFJet450' : 450,
            'HLT_PFJet500' : 500,
            'HLT_PFJet550' : 550,
        }
        
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
        ext.add_weight_sets([
            "* * data/jec/Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi.txt",
            "* * data/jec/Winter22Run3_V2_MC_L2Relative_AK4PFPuppi.txt",
            "* * data/jec/Winter22Run3_V2_MC_L3Absolute_AK4PFPuppi.txt",
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
        
        self._jet_factory_offline = CorrectedJetsFactory(name_map, jec_stack)
        
    def process(self, events):
        
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
        
        dataset = events.metadata['dataset']
        output = defaultdict()
        
        for trigger, X in self._triggers.items():
            
            h = Hist(
                hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
                hist.axis.StrCategory([], name="trigger", label="Trigger name", growth=True),
                hist.axis.Regular(100, 0, 1000, name="pt_s", label="Average scouting jet $p_T$"),
                hist.axis.Regular(100, 0, 1000, name="pt_o", label="Average offline jet $p_T$"),
            )
            
            split = trigger.split("_")
            start = split[0]
            rest = "_".join(split[1:])
            
            events_trigger = events[(events[start][rest])]
        
            jets_s = apply_jec(events_trigger.ScoutingJet, "ScoutingRho", events_trigger)
            pt_type = "pt_jec"
            jets_s = jets_s[
                (abs(jets_s.eta) < 2.5)
                & (jets_s.neHEF < 0.9)
                & (jets_s.neEmEF < 0.9)
                & (jets_s.muEmEF < 0.8)
                & (jets_s.chHEF > 0.01)
                #& (jets_s.nCh > 0)
                & (jets_s.chEmEF < 0.8)
            ]

            jets_o = apply_jec(events_trigger.Jet, "fixedGridRhoFastjetAll", events_trigger)
            jets_o = jets_o[
                (abs(jets_o.eta) < 2.5)
                & (jets_o.neHEF < 0.9)
                & (jets_o.neEmEF < 0.9)
                & (jets_o.muEF < 0.8)
                & (jets_o.chHEF > 0.01)
                #& (jets_o.nCh > 0)
                & (jets_o.chEmEF < 0.8)
            ]

            jets_ss = jets_s[
                (ak.num(jets_s) > 1)
                & (ak.num(jets_o) > 1)
            ]
            jets_oo = jets_o[
                (ak.num(jets_s) > 1)
                & (ak.num(jets_o) > 1)
            ]

            def require_back2back(obj1, obj2, phi=2.7):

                return (abs(obj1.delta_phi(obj2)) > phi)

            def require_3rd_jet(jets, pt_type="pt"):

                jet = jets[:, 2]
                pt_ave = (jets[:, 0][pt_type] + jets[:, 1][pt_type]) / 2

                return ((jet[pt_type] / pt_ave) < 0.3)

            def require_n(jets_s, jets_o, two=True):

                if two:
                    jet_s = jets_s[(ak.num(jets_s) == 2) & (ak.num(jets_o) == 2)][:, :2]
                    jet_o = jets_o[(ak.num(jets_s) == 2) & (ak.num(jets_o) == 2)][:, :2]
                else:
                    jet_s = jets_s[(ak.num(jets_s) > 2) & (ak.num(jets_o) > 2)]
                    jet_o = jets_o[(ak.num(jets_s) > 2) & (ak.num(jets_o) > 2)]

                return jet_s, jet_o

            def require_eta(jets):

                return ((abs(jets[:, 0].eta) < 1.3) | (abs(jets[:, 1].eta) < 1.3))

            def criteria_one(jet_s, jet_o, phi=2.7):

                b2b_s = require_back2back(jet_s[:, 0], jet_s[:, 1], phi)
                eta_s = require_eta(jet_s)

                b2b_o = require_back2back(jet_o[:, 0], jet_o[:, 1], phi)
                eta_o = require_eta(jet_o)

                req = ((b2b_s) & (b2b_o)) # & (eta_s) & (eta_o))

                return jet_s[req], jet_o[req]

            def criteria_n(jets_s, jets_o, pt_type="pt", phi=2.7):

                third_jet_s = require_3rd_jet(jets_s, pt_type)
                third_jet_o = require_3rd_jet(jets_o, pt_type)

                jet_s = jets_s[(third_jet_s) & (third_jet_o)][:, :2]
                jet_o = jets_o[(third_jet_s) & (third_jet_o)][:, :2]

                b2b_s = require_back2back(jet_s[:, 0], jet_s[:, 1], phi)
                eta_s = require_eta(jet_s)

                b2b_o = require_back2back(jet_o[:, 0], jet_o[:, 1], phi)
                eta_o = require_eta(jet_o)

                req = ((b2b_s) & (b2b_o)) # & (eta_s) & (eta_o))

                return jet_s[req], jet_o[req]

            def run_deltar_matching(obj1, obj2, radius=0.4): # NxM , NxG arrays
                _, obj2 = ak.unzip(ak.cartesian([obj1, obj2], nested=True)) # Obj2 is now NxMxG
                obj2['dR'] = obj1.delta_r(obj2)  # Calculating delta R
                t_index = ak.argmin(obj2.dR, axis=-2) # Finding the smallest dR (NxG array)
                s_index = ak.local_index(obj1.eta, axis=-1) #  NxM array
                _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True)) 
                obj2 = obj2[s_index == t_index] # Pairwise comparison to keep smallest delta R

                # Cutting on delta R
                obj2 = obj2[obj2.dR < radius] # Additional cut on delta R, now a NxMxG' array 
                return obj2

            def cut_ratio(ratio, h4=False):

                if h4:
                    return ratio, np.ones(len(ratio), dtype=bool)

                cut = (np.abs(ratio - 1) < 2) 

                return ratio[cut], cut

            jets_s_2, jets_o_2 = require_n(jets_ss, jets_oo, two=True)
            jets_s_n, jets_o_n = require_n(jets_ss, jets_oo, two=False)

            dijet_s_2, dijet_o_2 = criteria_one(jets_s_2, jets_o_2, phi=2.7)
            dijet_s_n, dijet_o_n = criteria_n(jets_s_n, jets_o_n, pt_type=pt_type, phi=2.7)

            dijet_2_dr = ak.flatten(run_deltar_matching(dijet_s_2, dijet_o_2, 0.2), axis=2)
            dijet_s_2_dr = dijet_s_2[(ak.num(dijet_2_dr) > 1)]
            dijet_o_2_dr = dijet_o_2[(ak.num(dijet_2_dr) > 1)]

            dijet_n_dr = ak.flatten(run_deltar_matching(dijet_s_n, dijet_o_n, 0.2), axis=2)
            dijet_s_n_dr = dijet_s_n[(ak.num(dijet_n_dr) > 1)]
            dijet_o_n_dr = dijet_o_n[(ak.num(dijet_n_dr) > 1)]

            for two in [True, False]:

                if (two):
                    dijet_s_dr = dijet_s_2_dr
                    dijet_o_dr = dijet_o_2_dr
                    n = "two"
                else:
                    dijet_s_dr = dijet_s_n_dr
                    dijet_o_dr = dijet_o_n_dr
                    n = "n"

                for ijet, jjet in [(0, 1), (1, 0)]:

                    tag_s = dijet_s_dr[:, ijet]
                    tag_o = dijet_o_dr[:, ijet]
                    probe_s = dijet_s_dr[:, jjet]
                    probe_o = dijet_o_dr[:, jjet]

                    # require HLT tag pt > X
                    req = (tag_s[pt_type] > X)
                    probe_s = probe_s[req]
                    probe_o = probe_o[req]

                    h.fill(
                        dataset = dataset,
                        trigger = "none",
                        pt_s = probe_s[pt_type],
                        pt_o = probe_o[pt_type]
                    )

                    # require HLT probe pt > X
                    req = (probe_s[pt_type] > X)
                    probe_s = probe_s[req]
                    probe_o = probe_o[req]

                    h.fill(
                        dataset = dataset,
                        trigger = trigger,
                        pt_s = probe_s[pt_type],
                        pt_o = probe_o[pt_type]
                    )
                    
            output[trigger] = h
                
        return output

    def postprocess(self, accumulator):
        pass
