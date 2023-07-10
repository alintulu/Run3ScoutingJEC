import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

from coffea import util
from hist import Hist
import hist

import mplhep
plt.style.use(mplhep.style.CMS)

import jer
import fit

colours = {
    "scouting" : "black", #"#FF66FF",
    "offline" : "red", #"#FF9900",
    "gen" : "#9A32CD",
}

colours_eta = {
    0 : "black", #"#FF66FF",
    1 : "red", #"#FF9900",
    sum : '#9A32CD',
}

markers = {
    "scouting" : "o",
    "offline" : "^",
    "gen" : "#9A32CD",
}

markers_eta = {
    0 : "o",
    1 : "^",
    sum : "#9A32CD",
}

data_err_opts = {
    'linestyle': 'none',
    'markersize': 10.,
    'elinewidth': 1,
}

eta = {
    0 : r"$|\eta| \leq 1.3$",
    1 : r"$1.3 < |\eta| \leq 2.5$",
    sum : r"$|\eta| \leq 2.5$",
}

def combine_triggers(output, is_data, is_chs, plot=True, response=False):

    ns = {
        "scouting" : [],
        "offline" : []
    }
    hs = defaultdict()
    hs["asymmetry" if not response else "response"] = {}
    
    commonaxes = (
        hist.axis.Variable([0, 85, 115, 145, 165, 210, 230, 295, 360, 445, 495, 550, 600] + list(np.arange(650, 2050, 50)), name="pt_ave", label=r"$p_{T}^{ave}$"),
        hist.axis.Variable([0, 0.01, 0.05, 0.1, 0.15, 0.2], name="alpha", label=r"$\alpha$"),
        hist.axis.StrCategory(["barrel", "endcap"], name="eta", label=r"|$\eta$|"),
    )
    
    if plot:    
        fig, axs = plt.subplots(9, 3, figsize=(30,60))
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        row0, column0, row1, column1  = 0, 0, 0, 0

    for rec in ns.keys():

        if rec == "offline": row0, column0 = row1, column1
        if rec == "scouting": row1, column1 = row0, column0

        h = Hist(
            *commonaxes,
            hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry") if not response else hist.axis.Regular(100, 0.5, 1.5, name="ratio", label=r"Reco $p_T$/Gen $p_T$"),
        )

        for i_trigger, (x_range, triggers) in enumerate(triggers_list.items()):
            trigger = triggers[0]

            if is_data and not is_chs:
                h_temp = output[trigger][rec if not response else rec + "_gen"][
                    hist.loc(x_range[0]):hist.loc(x_range[1]),
                    sum,
                    :,
                    :,
                    sum,
                    :
                ]
            else:
                h_temp = output[trigger][rec if not response else rec + "_gen"][
                    hist.loc(x_range[0]):hist.loc(x_range[1]),
                    :,
                    :,
                    sum,
                    :
                ]

            if trigger != "HLT_PFJet550":
                h_temp = h_temp[{"pt_ave" : 0}]
                h[i_trigger + 1, :, :, :] = h_temp.view()
            else:
                h[11:39, :, :, :] = h_temp.view()
                h_temp = h_temp[{"pt_ave" : sum}]

            h_i = h_temp[hist.loc(0):hist.loc(0.1):sum, sum, :]
            N = h_i.sum()
            ns[rec].append(N)

            if plot:
                h_i *= 1 / N if N > 0 else 1
                ax = axs[row0, column0]
                mplhep.histplot(h_i, ax=ax, label=rec, color=colours[rec])
                ax.set_title(f"{x_range[0]}" 
                             + r"< $p_T^{ave} <$" 
                             + f"{x_range[1]}" + r" (|$\eta$| < 2.5, $\alpha$" + f"< {0.1})"
                             , fontsize=23)
                ax.legend(loc="upper left", title=trigger)
                if response: ax.set_xlabel(r"Reco $p_T$ / Gen $p_T$") 

            column0 += 1
            if column0 > 2:
                column0 = 0
                row0 += 1

        hs["asymmetry" if not response else "response"][rec] = h

    if plot:
        for i, ax in enumerate(axs.flat):
            if not bool(ax.has_data()):
                fig.delaxes(ax)

        fig.savefig(
           f'step0/{"data" if is_data else "mc"}/triggers_{"asymmetry" if not response else "response"}.png',
           bbox_inches='tight'
        )

    with open(f'step0/{"data" if is_data else "mc"}/ns{"" if not response else "_response"}.pickle', "wb") as handle:
            pickle.dump(hs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    if not is_data and response:
        with open(f'step0/mc/ns.pickle', 'rb') as handle:
            asymmetry = pickle.load(handle)
        with open(f'step0/mc/ns_response.pickle', 'rb') as handle:
            response = pickle.load(handle)
            
        combine = {
            "asymmetry" : asymmetry["asymmetry"],
            "response" : response["response"],
        }
        
        with open(f'step0/mc/ns.pickle', "wb") as handle:
            pickle.dump(combine, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.remove(f'step0/mc/ns_response.pickle')
        
def plot_pu(output):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    reweight = output["pu"][{"isreweight" : hist.loc("True")}][hist.rebin(2)]
    reweight *= 1 / reweight.sum() if reweight.sum() > 0 else 1
    notreweight = output["pu"][{"isreweight" : hist.loc("False")}][hist.rebin(2)]
    notreweight *= 1 / notreweight.sum() if notreweight.sum() > 0 else 1
    
    mplhep.histplot(notreweight, ax=ax, label="Before pileup\nreweighting", color=colours_eta[0])
    mplhep.histplot(reweight, ax=ax, label="After pileup\nreweighting\n" + r"(<$\mu$> =" + f" {jer.mean(reweight):.0f})", color=colours_eta[1])
    
    label = jer.set_title(is_data, ax)
    xlabel = ax.set_xlabel("Number of pileup interactions")
    ax.set_ylabel("Normalised to unit")
    ax.set_xlim(-5, 105)
    ax.legend(loc="upper left")

    fig.savefig(
        f'step0/mc/pu.png',
        bbox_extra_artists=(label[0], label[1], xlabel),
        bbox_inches='tight'
    )
    
def compute_response(output, rebin, plot=True):
    
    results = {}
    
    for ieta in [0, 1]:

        if plot:
            fig, axs = plt.subplots(9, 3, figsize=(30,60))
            plt.subplots_adjust(wspace=0.2, hspace=0.5)
            row0, column0, row1, column1 = 0, 0, 0, 0
            
            fig1, ax1 = plt.subplots(figsize=(10,10))
            
        eta_region = {
            0 : "barrel",
            1 : "endcap",
        }
    
        result = {
            "scouting" : {
               "center" : [],
               "edge" : [],
               "resp" : [],
               "resp_unc" : [], 
            },
            "offline" : {
               "center" : [],
               "edge" : [],
               "resp" : [],
               "resp_unc" : [], 
            },  
        }
            
        for (lbin, rbin), nbin in rebin.items():

            for rec in ["scouting", "offline"]:

                if rec == "offline": row0, column0 = row1, column1
                if rec == "scouting": row1, column1 = row0, column0
                    
                if rec == "scouting":
                    h = output['Run3Summer22EE'][rec][
                            hist.loc(lbin):hist.loc(rbin):nbin, # pt
                            hist.loc(eta_region[ieta]),         # eta
                            :,                                  # ratio
                            hist.loc('0.2'),                    # match
                            sum,                                # alpha
                            hist.loc('True')                    # dijet
                        ]
                else:
                    h = output['Run3Summer22EE'][rec][
                            hist.loc(lbin):hist.loc(rbin):nbin, # pt
                            hist.loc(eta_region[ieta]),         # eta
                            :,                                  # ratio
                            hist.loc('0.2'),                    # match
                            sum,                                # alpha
                            hist.loc('True')                    # dijet
                        ]

                centers = h.axes[0].centers
                edges = h.axes[0].edges

                for i, center in enumerate(centers):
                    
                    if (center < 80 and nbin == 1j) or (center > 1000 and ieta == 1):
                        continue
                    
                    ax = axs[row0, column0]

                    h_i = h[{"pt_ave" : i}]

                    median, median_unc = jer.get_median(h_i.axes[0].centers, h_i.values(), h_i.axes[0].edges, h_i.sum())
                    
                    result[rec]["center"].append(center)
                    result[rec]["edge"].append((edges[i], edges[i+1]))
                    result[rec]["resp"].append(median)
                    result[rec]["resp_unc"].append(median_unc)
                    
                    closure = abs(median - 1) * 100 if median > 0 else 0

                    if plot:
                        h_i *= 1 / h_i.sum() if h_i.sum() > 0 else 1
                        
                        mplhep.histplot(h_i, ax=ax, color=colours[rec], label=f"{rec.capitalize()} (Closure: {closure:.3f} %)")
                        ax.legend(loc="upper left")
                        ax.set_title(f"{edges[i]}" + r" < Gen $p_T < $" + f"{edges[i+1]}", fontsize=23)
                        ax.set_xlabel(r"Reco $p_T$ / Gen $p_T$")

                        column0 += 1
                        if column0 > 2:
                            column0 = 0
                            row0 += 1

        results[ieta] = result
                            
        if plot:
            for i, ax in enumerate(axs.flat):
                if not bool(ax.has_data()):
                    fig.delaxes(ax)

            fig.savefig(
                    f'step1/mc/closure_{"barrel" if ieta == 0 else "endcap"}.png',
                    bbox_inches='tight'
                )
            for rec in ["scouting", "offline"]:
                
                x = result[rec]["center"]
                xerr = result[rec]["edge"]
                y = np.array(result[rec]["resp"])
                
                ax1.errorbar(
                    x,
                    y,
                    xerr=[
                        [center - edge[0] for center, edge in zip(x, xerr)],
                        [edge[1] - center for center, edge in zip(x, xerr)]
                    ],
                    yerr=result[rec]["resp_unc"],
                    color=colours[rec],
                    marker=markers[rec],
                    **data_err_opts,
                    label=rec.capitalize()
                )
            
            ax1.set_ylim(0.96, 1.04)
            ax1.set_xlim(0, 2000)
            ax1.axhline(1, color="gray", linestyle="--")
            label = jer.set_title(is_data, ax1)
            ax1.legend(loc="best", title=eta[ieta])
            ax1.set_ylabel(r"Response")
            ax1.set_xlabel(r"Gen $p_T$ (GeV)")
            
            fig1.savefig(
                    f'step1/mc/response_{"barrel" if ieta == 0 else "endcap"}.pdf',
                    bbox_inches='tight'
                )
            
    with open(f'step1/mc/resp.pickle', "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compute_jer(is_data, is_chs, rebin, plot=True, alphas=[0.05, 0.1, 0.15, 0.2]):

    res = defaultdict()

    with open(f'step0/{"data" if is_data else "mc"}/ns.pickle', 'rb') as handle:
        hs = pickle.load(handle)
 
    for ieta in [0, 1]:
        
        res[ieta] = {}
        for rec in ["scouting", "offline"]:
            res[ieta][rec] = {}
            for alpha in alphas:
                res[ieta][rec][alpha] = {
                                "center" : [],
                                "edge" : [],
                                "n" : [],
                                "sigma" : [],
                                "sigma_unc" : [],
                                "mu" : [],
                                "mu_unc" : [],
                            }
        if plot:    
            fig, axs = plt.subplots(9, 3, figsize=(30,60))
            plt.subplots_adjust(wspace=0.2, hspace=0.5)
            row0, column0, row1, column1  = 0, 0, 0, 0
            
            
        for alpha in alphas:

            for (lbin, rbin), nbin in rebin.items():

                h1 = hs["asymmetry"]["scouting"][
                        hist.loc(lbin):hist.loc(rbin):nbin,
                        hist.loc(0):hist.loc(alpha):sum,
                        ieta,
                        hist.rebin(1)
                    ]

                h2 = hs["asymmetry"]["offline"][
                        hist.loc(lbin):hist.loc(rbin):nbin,
                        hist.loc(0):hist.loc(alpha):sum,
                        ieta,
                        hist.rebin(1)
                    ]

                centers = h1.axes[0].centers
                edges = h1.axes[0].edges

                for i, center in enumerate(centers):

                    if (center < 80 and nbin == 1j) or (center > 1000 and ieta == 1):
                        continue
                        
                    for h, rec in [(h1, "scouting"), (h2, "offline")]:

                        h_i = h.copy()[{"pt_ave" : i}]
                        N = h_i.sum()
                        if N < 1:
                            continue

                        rec1 = rec

                        plot_format = {
                            'linestyle' : 'solid' if rec1 == "scouting" else 'dashed',
                            'linewidth' : 2,
                        }
                        
                        mu, mu_unc = jer.get_median(h_i.axes[0].centers, h_i.values(), h_i.axes[0].edges, h_i.sum())
                        res[ieta][rec1][alpha]["center"].append(center)
                        res[ieta][rec1][alpha]["edge"].append((edges[i], edges[i+1]))
                        res[ieta][rec1][alpha]["n"].append(N)
                        res[ieta][rec1][alpha]["sigma"].append(jer.confidence(h_i))
                        res[ieta][rec1][alpha]["sigma_unc"].append(jer.confidence_unc(jer.confidence(h_i), N))
                        res[ieta][rec1][alpha]["mu"].append(mu)
                        res[ieta][rec1][alpha]["mu_unc"].append(mu_unc)
                        h_i *= 1 / N if N != 0 else 1

                        if plot and alpha == 0.05:
                            ax = axs[row0, column0]
                            
                            plot_axs = [(ax, fig, False)]
                            
                            if (ieta == 0 and center == 600.0):
                                if rec == "scouting": fig_big, ax_big = plt.subplots(figsize=(10,10))
                                plot_axs = [(ax, fig, False), (ax_big, fig_big, True)]

                            for ax1, fig1, big in plot_axs:
                                mplhep.histplot(
                                    h_i,
                                    ax=ax1,
                                    label=rec1.capitalize(),
                                    color=colours[rec1],
                                    **plot_format
                                )
                                
                                if big and rec == "offline":
                                    label = jer.set_title(is_data, ax1)
                                    ax_big.set_ylabel("Normalised to unit")
                                    ax_big.set_xlabel(r"Asymmetry $(p_T^1 - p_T^2) / (p_T^1 + p_T^2)$")
                                    ax_big.set_ylim(0, 0.1)
                                    ax_big.legend(loc="upper left",
                                                  title=f"{edges[i]:.0f}" + r" < $p_T^{ave} < $" + f"{edges[i+1]:.0f} GeV" + "\n" + f"{eta[ieta]}")
                                    fig1.savefig(
                                        f'step1/{"data" if is_data else "mc"}/hist_asymmetry_{"barrel" if ieta == 0 else "endcap"}_{centers[i]:.0f}.pdf',
                                        bbox_extra_artists=(label[0], label[1]),
                                        bbox_inches='tight'
                                    )
                                
                            ax.set_title(f"{edges[i]}" 
                                         + r" < $p_T^{ave} < $" 
                                         + f"{edges[i+1]} ({eta[ieta]}, " + r"$\alpha$ < " + f"{alpha})"
                                         , fontsize=23)
                    if plot and alpha == 0.05:        
                        h, l = ax.get_legend_handles_labels()
                        i_labels = np.array(l).argsort()[::-1]
                        ax.legend([h[idx] for idx in i_labels],[l[idx] for idx in i_labels], loc="upper left")

                    column0 += 1
                    if column0 > 2:
                        column0 = 0
                        row0 += 1

        if plot:
            for i, ax in enumerate(axs.flat):
                if not bool(ax.has_data()):
                    fig.delaxes(ax)
                    continue
                N_text = "No. of events: s, o\n"
                for _, alpha in enumerate(alphas):                    
                    temp_s = round(res[ieta]["scouting"][alpha]["n"][i])
                    temp_o = round(res[ieta]["offline"][alpha]["n"][i])
                    N_text += r"$\alpha<$" + f"{alpha}: {temp_s}, {temp_o}\n"

                    ax.text(0.95, 0.9, N_text,
                        fontsize=18,
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax.transAxes
                   )

        fig.savefig(
            f'step1/{"data" if is_data else "mc"}/hist_asymmetry_{"barrel" if ieta == 0 else "endcap"}.pdf',
            bbox_inches='tight'
        )
        
    with open(f'step1/{"data" if is_data else "mc"}/res.pickle', "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def plot_jer(is_data):
    
    if os.path.exists(f'step1/{"data" if is_data else "mc"}/res.pickle'):
        with open(f'step1/{"data" if is_data else "mc"}/res.pickle', 'rb') as handle:
            res = pickle.load(handle)
    else:
        raise ValueError(f'step1/{"data" if is_data else "mc"}/res.pickle does not exist')
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    for ax, ieta in [(ax1, 0)]:
        
        for rec in ["scouting", "offline"]:
            
            centers = res[ieta][rec][0.05]["center"]
            y = res[ieta][rec][0.05]["sigma"]
            yerr = res[ieta][rec][0.05]["sigma_unc"]

            ax.errorbar(
                centers,
                y,
                xerr=jer.get_xerr(centers),
                yerr=yerr,
                color=colours[rec],
                **data_err_opts,
                label=rec.capitalize()
            )
            
        label = jer.set_title(is_data, ax)
        ax.legend(loc="best", title=eta[ieta].capitalize() + r" ($\alpha$ < " + f"{0.05})")
        xlabel = ax.set_xlabel(r"$p_T^{ave}$ (GeV)")
        ax.set_ylabel(r"$\sigma_A$")
        #ax.set_ylim(0, 0.35)
        ax.set_xlim(0, 2000)
        
    fig.savefig(
        f'step1/{"data" if is_data else "mc"}/sigma.pdf',
        bbox_extra_artists=(label[0], label[1], xlabel),
        bbox_inches='tight'
    )
        
def plot_ratio(is_data):        

    if os.path.exists(f'step1/{"data" if is_data else "mc"}/res.pickle'):
        with open(f'step1/{"data" if is_data else "mc"}/res.pickle', 'rb') as handle:
            res = pickle.load(handle)
    else:
        raise ValueError(f'step1/{"data" if is_data else "mc"}/res.pickle does not exist')
        
    fig, ax = plt.subplots(figsize=(10, 10))
    for ieta in [0, 1]:
        
        centers = res[ieta]["scouting"][0.05]["center"]
        y1 = res[ieta]["scouting"][0.05]["sigma"]
        y2 = res[ieta]["offline"][0.05]["sigma"]
        yerr = jer.err_prop(
            res[ieta]["scouting"][0.05]["sigma"],
            res[ieta]["offline"][0.05]["sigma"],
            res[ieta]["scouting"][0.05]["sigma_unc"],
            res[ieta]["offline"][0.05]["sigma_unc"],
        )

        ax.errorbar(
            centers,
            np.array(y1) / np.array(y2),
            xerr=jer.get_xerr(centers),
            yerr=yerr,
            color=colours_eta[ieta],
            **data_err_opts,
            label=eta[ieta],
        )
            
    label = jer.set_title(is_data, ax)
    ax.legend(loc="best", title=eta[ieta].capitalize() + r" ($\alpha$ < " + f"{0.05})")
    ax.set_xlabel(r"$p_T^{ave}$ (GeV)")
    xlabel = ax.set_ylabel(r"$\sigma_A$")
    ax.set_ylim(0.7, 1.4)
    ax.set_xlim(0, 2000)
    ax.axhline(1, color="gray", linestyle="--")
        
    fig.savefig(
        f'step1/{"data" if is_data else "mc"}/ratio_sigma.pdf',
        bbox_extra_artists=(label[0], label[1], xlabel),
        bbox_inches='tight'
    )
    
def plot_jes(is_data, alpha=0.1):
    
    if os.path.exists(f'step1/{"data" if is_data else "mc"}/res.pickle'):
        with open(f'step1/{"data" if is_data else "mc"}/res.pickle', 'rb') as handle:
            res = pickle.load(handle)
    else:
        raise ValueError(f'step1/{"data" if is_data else "mc"}/res.pickle does not exist')
        
    for ieta in [0, 1]:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for rec in ["scouting", "offline"]:
        
            centers = res[ieta][rec][alpha]["center"]
            xerr = res[ieta]["scouting"][alpha]["edge"]
            y = res[ieta][rec][alpha]["mu"]
            yerr = res[ieta][rec][alpha]["mu_unc"]

            ax.errorbar(
                centers,
                y,
                xerr=[
                    [center - edge[0] for center, edge in zip(centers, xerr)],
                    [edge[1] - center for center, edge in zip(centers, xerr)]
                ],
                yerr=yerr,
                color=colours[rec],
                **data_err_opts,
                label=rec.capitalize()
            )
            
        label = jer.set_title(is_data, ax)
        ax.legend(loc="lower right", title=eta[ieta])
        xlabel = ax.set_xlabel(r"Average $p_T$ (GeV)")
        ax.set_ylabel(r"$\mu_A$")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylim(-0.03, 0.03)
        ax.set_xlim(0, 2000)
        
        fig.savefig(
            f'step1/{"data" if is_data else "mc"}/mu_{"barrel" if ieta == 0 else "endcap"}.pdf',
            bbox_extra_artists=(label[0], label[1], xlabel),
            bbox_inches='tight'
        )
        
    fig, ax = plt.subplots(figsize=(10, 10))
        
    for ieta in [0, 1]:
        
        centers = res[ieta]["scouting"][alpha]["center"]
        xerr = res[ieta]["scouting"][alpha]["edge"]
        y1 = res[ieta]["scouting"][alpha]["mu"]
        y2 = res[ieta]["offline"][alpha]["mu"]
        yerr1 = res[ieta]["scouting"][alpha]["mu_unc"]
        yerr2 = res[ieta]["offline"][alpha]["mu_unc"]

        ax.errorbar(
            centers,
            (np.array(y1) + 1) / (np.array(y2) + 1),
            xerr=[
                [center - edge[0] for center, edge in zip(centers, xerr)],
                [edge[1] - center for center, edge in zip(centers, xerr)]
            ],
            yerr=jer.err_prop(
                (np.array(y1) + 1),
                (np.array(y2) + 1),
                yerr1,
                yerr2,
            ),
            color=colours_eta[ieta],
            **data_err_opts,
            label=eta[ieta]
       )
            
    label = jer.set_title(is_data, ax)
    ax.legend(loc="lower right")
    xlabel = ax.set_xlabel(r"Average $p_T$ (GeV)")
    ax.set_ylabel(r"$(\mu_A^{Scouting} + 1) / (\mu_A^{Offline} + 1)$")
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_ylim(0.96, 1.04)
    ax.set_xlim(0, 2000)

    fig.savefig(
        f'step1/{"data" if is_data else "mc"}/ratio_mu.pdf',
        bbox_extra_artists=(label[0], label[1], xlabel),
        bbox_inches='tight'
    )
    
def alpha_extrapolate(is_data, plot=True, response=False, alphas=[0.05, 0.1, 0.15, 0.2]):
    
    var = "sigma"
    
    if os.path.exists(f'step1/{"data" if is_data else "mc"}/res.pickle'):
        with open(f'step1/{"data" if is_data else "mc"}/res.pickle', 'rb') as handle:
            res = pickle.load(handle)
    else:
        raise ValueError(f'step1/{"data" if is_data else "mc"}/res.pickle does not exist')
     
    if response:
        with open('step1/mc/resp.pickle', 'rb') as handle:
            resp = pickle.load(handle)
        
    for ieta in [0, 1]:
        
        if plot:    
            fig, axs = plt.subplots(9, 3, figsize=(30,60))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            row, column = 0, 0
            
        centers = res[ieta]["scouting"][0.05]["center"] 
        for rec in ["scouting", "offline"]:
            res[ieta][rec][0] = {
                var : [],
                f"{var}_unc" : [],
            }

        for i_center, center in enumerate(centers):

            x1 = alphas
            y1 = [res[ieta]["scouting"][alpha][var][i_center] for alpha in alphas]
            yerr1 = [res[ieta]["scouting"][alpha][f"{var}_unc"][i_center] for alpha in alphas]
            if response:
                denom1 = resp[ieta]["scouting"]["resp"][i_center]
                y1 = [y / denom1 for y in y1]
            
            y2 = [res[ieta]["offline"][alpha][var][i_center] for alpha in alphas]
            yerr2 = [res[ieta]["offline"][alpha][f"{var}_unc"][i_center] for alpha in alphas]
            if response:
                denom2 = resp[ieta]["offline"]["resp"][i_center]
                y2 = [y / denom2 for y in y2]

            if plot:
                ax = axs[row, column]
                
            for x, y, yerr, rec in [(x1, y1, yerr1, "scouting"), (x1, y2, yerr2, "offline")]:
                
                if is_data:
                    # remove outliers that has low stastistics
                    remove_point = {
                        "offline" : {
                            1 : {
                                0 : [0, 2, 3],
                            },
                        },
                        "scouting" : {
                            0 : {
                                0 : [1, 2, 3],
                            },
                        }
                    }
                    
                x_new = x.copy()
                y_new = y.copy()
                yerr_new = yerr.copy()
                
                if rec in remove_point:
                    if ieta in remove_point[rec]:
                        if i_center in remove_point[rec][ieta]:
                            x_new = [x_new[i] for i in remove_point[rec][ieta][i_center]]
                            y_new = [y_new[i] for i in remove_point[rec][ieta][i_center]]
                            yerr_new = [yerr_new[i] for i in remove_point[rec][ieta][i_center]]
                
                plot_axs = [(ax, fig, False)]
                if (ieta == 0 and i_center == 4):
                    if rec == "scouting": fig_big, ax_big = plt.subplots(figsize=(10,10))
                    plot_axs = [(ax, fig, False), (ax_big, fig_big, True)]
                    
                edge1, edge2 = res[ieta]["scouting"][0.05]["edge"][i_center]
                ax.set_title(f"{edge1}" + r" < $p_{T}^{ave}$ < " + f"{edge2} ({eta[ieta]})", fontsize=23)
                
                for ax1, fig1, big in plot_axs:
                    m0, sm0, c0, sc0, cov0, rho0 = fit.lin_fit(
                      ax1,
                      np.array(x_new),
                      np.array(y_new),
                      np.array(yerr_new),
                      r"$\alpha$",
                      r"$\sigma_A$" if not response else r"$\sigma_A/Response$",
                      0,
                      0.25,
                      min(min(y1), min(y2)) - 0.05,
                      max(max(y1), max(y2)) + 0.05,
                      plot=plot, 
                      label=rec.capitalize(),
                      title="" if not big else f"{edge1:.0f}" + r" < $p_{T}^{ave}$ < " + f"{edge2:.0f} GeV" + "\n" + f"{eta[ieta]}",
                      colour=colours[rec],
                      setrange=True
                    )
                    
                    if big and rec == "offline":
                        ax.set_xlabel("")
                        if not response: ax1.set_ylim(0.08, 0.16)
                        ax1.set_xticklabels([r'$\alpha$ = 0', r'$\alpha < 0.05$', r'$\alpha < 0.1$', r'$\alpha < 0.15$', r'$\alpha < 0.2$', r'$\alpha < 0.25$'])
                        label = jer.set_title(is_data, ax1)
                        fig1.savefig(
                            f'step2/{"data" if is_data else "mc"}/fit_resolution_{"barrel" if ieta == 0 else "endcap"}_{centers[i_center]:.0f}.pdf',
                            bbox_extra_artists=(label[0], label[1]),
                            bbox_inches='tight'
                        )

                res[ieta][rec][0][var].append(c0)
                res[ieta][rec][0][f"{var}_unc"].append(sc0)

            if plot:
                column += 1
                if column > 2:
                    column = 0
                    row += 1

        if plot:
            for i, ax in enumerate(axs.flat):
                if not bool(ax.has_data()):
                    fig.delaxes(ax)
            fig.savefig(
                f'step2/{"data" if is_data else "mc"}/fit_resolution_{"barrel" if ieta == 0 else "endcap"}.png',
                bbox_inches='tight'
            )

    if plot:
        for ieta in [0, 1]:
            fig, ax = plt.subplots(figsize=(10, 5))

            for rec in ["scouting", "offline"]:
                
                x = res[ieta]["scouting"][0.05]["center"]
                xerr = res[ieta]["scouting"][0.05]["edge"]
                y = res[ieta][rec][0][var]
                yerr = res[ieta][rec][0][f"{var}_unc"]

                ax.errorbar(
                    x,
                    np.array(y) * np.sqrt(2),
                    xerr=[
                        [center - edge[0] for center, edge in zip(x, xerr)],
                        [edge[1] - center for center, edge in zip(x, xerr)]
                    ],
                    yerr=yerr,
                    color=colours[rec],
                    marker=markers[rec],
                    **data_err_opts,
                    label=rec.capitalize(),
                )
                
                label = jer.set_title(is_data, ax)
                xlabel = ax.set_xlabel(r"Average $p_T$ (GeV)")
                ax.set_ylabel(r"JER")
                ax.legend(loc="upper right", title=eta[ieta])
                ymin, ymax = (0, 0.2)
                ax.set_ylim(ymin, ymax)
#                 ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
                ax.set_xlim(0, 2000)
                
            fig.savefig(
                f'step2/{"data" if is_data else "mc"}/resolution_{"barrel" if ieta == 0 else "endcap"}.pdf',
                bbox_extra_artists=(label[0], label[1], xlabel),
                bbox_inches='tight'
            )
            
        fig, ax = plt.subplots(figsize=(10, 10))
        for ieta in [0, 1]:

            x = res[ieta]["scouting"][0.05]["center"]
            xerr = res[ieta]["scouting"][0.05]["edge"]
            y = np.array(res[ieta]["scouting"][0][var]) / np.array(res[ieta]["offline"][0][var])
            yerr = jer.err_prop(
                res[ieta]["scouting"][0][var],
                res[ieta]["offline"][0][var],
                res[ieta]["scouting"][0][f"{var}_unc"],
                res[ieta]["offline"][0][f"{var}_unc"]
            )

            ax.errorbar(
                x,
                y,
                xerr=[
                    [center - edge[0] for center, edge in zip(x, xerr)],
                    [edge[1] - center for center, edge in zip(x, xerr)]
                ],
                yerr=yerr,
                color=colours_eta[ieta],
                marker=markers_eta[ieta],
                **data_err_opts,
                label=eta[ieta],
            )

            label = jer.set_title(is_data, ax)
            xlabel = ax.set_xlabel(r"Average $p_T$ (GeV)")
            ax.set_ylabel(r"$JER^{Scouting} / JER^{Offline}$")
            ax.legend(loc="upper right", fontsize=28)
#                 ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
            ax.set_ylim(0.7, 1.4)
            ax.set_xlim(0, 2000)
            ax.axhline(1, color="gray", linestyle="--")

        fig.savefig(
            f'step2/{"data" if is_data else "mc"}/ratio_resolution.pdf',
            bbox_extra_artists=(label[0], label[1], xlabel),
            bbox_inches='tight'
        )

    with open(f'step2/{"data" if is_data else "mc"}/res.pickle', "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_data_vs_mc():
    
    ress = defaultdict()
    
    for sample in ["data", "mc"]:
        if os.path.exists(f'step2/{sample}/res.pickle'):
            with open(f'step2/{sample}/res.pickle', 'rb') as handle:
                res = pickle.load(handle)
            ress[sample] = res
        else:
            raise ValueError(f'step2/{sample}/res.pickle does not exist')
            
    for ieta in [0, 1]:
        
        fig, ax = plt.subplots(figsize=(10, 10))
        for rec in ["scouting", "offline"]:

            x = ress["mc"][ieta][rec][0.05]["center"]
            xerr = ress["mc"][ieta]["scouting"][0.05]["edge"]
            y = np.array(ress["data"][ieta][rec][0]["sigma"]) / np.array(ress["mc"][ieta][rec][0]["sigma"])
            yerr = jer.err_prop(
                ress["data"][ieta][rec][0]["sigma"],
                ress["mc"][ieta][rec][0]["sigma"],
                ress["data"][ieta][rec][0]["sigma_unc"],
                ress["mc"][ieta][rec][0]["sigma_unc"],
            )

            ax.errorbar(
                x,
                y,
                xerr=[
                    [center - edge[0] for center, edge in zip(x, xerr)],
                    [edge[1] - center for center, edge in zip(x, xerr)]
                ],
                yerr=yerr,
                color=colours[rec],
                marker=markers[rec],
                **data_err_opts,
                label=rec.capitalize(),
            )

        label = jer.set_title(is_data, ax)
        xlabel = ax.set_xlabel(r"Average $p_T$ (GeV)")
        ax.set_ylabel(r"$JER^{Data} / JER^{Simulation}$")
        ax.legend(loc="lower right", title=eta[ieta])
#         ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
        ax.set_ylim(0.7, 1.4)
        ax.set_xlim(0, 2000)
        ax.axhline(1, color="gray", linestyle="--")

        fig.savefig(
            f'step3/ratio_{"barrel" if ieta == 0 else "endcap"}.pdf',
            bbox_extra_artists=(label[0], label[1], xlabel),
            bbox_inches='tight'
        )
        
def plot_jes_ratio(alpha=0.1):
    
    ress = defaultdict()
    
    for sample in ["data", "mc"]:
        if os.path.exists(f'step1/{sample}/res.pickle'):
            with open(f'step1/{sample}/res.pickle', 'rb') as handle:
                res = pickle.load(handle)
            ress[sample] = res
        else:
            raise ValueError(f'step1/{sample}/res.pickle does not exist')
    
    for ieta in [0, 1]:
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for rec in ["scouting", "offline"]:
        
            centers = ress["mc"][ieta]["scouting"][alpha]["center"]
            xerr = ress["mc"][ieta]["scouting"][alpha]["edge"]
            y1 = ress["data"][ieta][rec][alpha]["mu"]
            y2 = ress["mc"][ieta][rec][alpha]["mu"]
            yerr1 = ress["data"][ieta][rec][alpha]["mu_unc"]
            yerr2 = ress["mc"][ieta][rec][alpha]["mu_unc"]

            ax.errorbar(
                centers,
                (np.array(y1) + 1) / (np.array(y2) + 1),
                xerr=[
                    [center - edge[0] for center, edge in zip(centers, xerr)],
                    [edge[1] - center for center, edge in zip(centers, xerr)]
                ],
                yerr=jer.err_prop(
                    (np.array(y1) + 1),
                    (np.array(y2) + 1),
                    yerr1,
                    yerr2,
                ),
                color=colours[rec],
                **data_err_opts,
                label=rec.capitalize()
            )

        label = jer.set_title(is_data, ax)
        ax.legend(loc="best", title=eta[ieta])
        xlabel = ax.set_xlabel(r"Average $p_T$ (GeV)")
        ax.set_ylabel(r"$(\mu_A^{Data} + 1) / (\mu_A^{Simulation} + 1)$")
        ax.axhline(1, color="gray", linestyle="--")
        ax.set_ylim(0.96, 1.04)
        ax.set_xlim(0, 2000)

        fig.savefig(
            f'step3/ratio_jes_{"barrel" if ieta == 0 else "endcap"}.pdf',
            bbox_extra_artists=(label[0], label[1], xlabel),
            bbox_inches='tight'
        )

triggers_list = {
    (85, 115) : ['L1_SingleJet60', 'HLT_PFJet60', 'HLT_PFJet80'],
    (115, 145) : ['L1_SingleJet90'],
    (145, 165) : ['L1_SingleJet120'],
    (165, 210) : ['HLT_PFJet140'],
    (210, 230) : ['L1_SingleJet180'],
    (230, 295) : ['L1_SingleJet200', 'HLT_PFJet200'],
    (295, 360) : ['HLT_PFJet260'],
    (360, 445) : ['HLT_PFJet320'],
    (445, 495) : ['HLT_PFJet400'], 
    (495, 550) : ['HLT_PFJet450'],
    (550, 600) : ['HLT_PFJet500'],
    (600, 2000) : ['HLT_PFJet550'],
}

# rebin = {
#     (85, 210): 4j,
#     (210, 650) : 2j,
#     (650, 2000) : 4j
# }

rebin = {
    (85, 210): 4j,
    (210, 295): 2j,
    (295, 650) : 2j,
    (650, 2000) : 4j
}

files = {
    "data" : {
        "chs" : "../outfiles/2022/jer_dijet_ScoutingPFMonitor_2022-CHS_offlineCHS_v2.coffea",
        "puppi" : "../outfiles/2022/prod/offlinePUPPI/jer_dijet_ScoutingPFMonitor_2022-CHS_offlinePuppi_v2.coffea",
    },
    "mc" :{
        "chs" : "input/jer_dijet_QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8_offlineCHS_v2.coffea",
        "puppi" : "input/jer_dijet_QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8_offlinePuppi_v2.coffea",
    },   
}
            
is_chs = False

for is_data, steps in [
    (False, {0, 1, 2}), 
    (True, {0, 1, 2, 3}),
    ]:
    
    output = util.load(files["data" if is_data else "mc"]["chs" if is_chs else "puppi"])[0]

    for step in steps:
        temp = "" if step == 3 else f'/{"data" if is_data else "mc"}'
        if not os.path.exists(f'step{step}{temp}'):
            print(f'Creating output directory step{step}{temp}...')
            os.makedirs(f'step{step}{temp}')

    ##### step 0
    if 0 in steps:
        combine_triggers(output, is_data, is_chs, triggers_list)
        if not is_data:
            combine_triggers(output, is_data, is_chs, triggers_list, response=True)
            plot_pu(output)

    #### step 1
    if 1 in steps:
        if not is_data:
            response = util.load("input/response_QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8.coffea")[0]
            compute_response(response, rebin)
        compute_jer(is_data, is_chs, rebin)
        plot_jes(is_data, alpha=0.15)

    #### step 2
    if 2 in steps:
        alpha_extrapolate(is_data, response=True)

    #### step 3
    if 3 in steps:
        plot_data_vs_mc()
        plot_jes_ratio(alpha=0.15)
