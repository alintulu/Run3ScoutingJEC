_The code was developed by Adelina Lintuluoto_

This README documents:

1. [Directory structure](#directory-structure), the contents of this Github repository
2. [Setup](#setup), instructions for creating the correct software environment to run the code
3. [Trigger selection](#trigger-selection), details on how to the trigger selection works
4. [JES](#jes), details on how the JES measurment works
5. [JER](#jer), details on how the JER measurment works

# Directory structure

This repository contains code developed with Coffea 0.7.21 for the analysis of jet energy corrections. The results of this analysis were reported in the DPS note:

- Jet energy scale and resolution measurements using data scouting events collected by the CMS experiment in 2022 at √s = 13.6 TeV: [CMS DP-2023/072](https://cds.cern.ch/record/2871592) ([twiki](https://twiki.cern.ch/twiki/bin/view/CMSPublic/Run3JESJERDataScouting))

The 5 folders of this repository contain:

- [data](data), the input data
- [jer](jer), code for analysing the JER
- [jes](jes), code for analysing the JES
- [processors](processors), the Coffea 0.7 processors used to create histograms
- [trigger](trigger), code for analysing trigger efficiencies

# Setup

1. Log into lpc
2. Follow [these](https://github.com/CoffeaTeam/lpcjobqueue) instructions to install `lpcjobqueue`
3. Make sure you have a valid grid proxy
4. Source an environment containing Coffea 0.7.21, e.g.

```
./shell coffeateam/coffea-dask:0.7.21-fastjet-3.4.0.1-ge327427
```

# Trigger selection

In order to avoid a trigger bias, events part of the JES and JER measurement are selected following a method outlined [here (page 4)](https://indico.cern.ch/event/740972/contributions/3059262/attachments/1678616/2696013/20180701_DijetHLT_efficiency_Run2018A.pdf#page=5) by Anastasia Karavdina. The implementation present in this code repository was presented at the JERC meeting [here (page 3-4)](https://indico.cern.ch/event/1304645/contributions/5496203/attachments/2683044/4654734/2023%2007%2012%20JER%20from%20dijet%20sample%20with%20new%20trigger%20strategy.pdf#page=3). The implementation is now discussed in detail.

Events are collected using an array of L1T seeds and HLT triggers (listed [here](https://github.com/alintulu/Run3ScoutingJEC/blob/master/processors/trigger.py#L24-L38)) that select events containing at least one jet with a $p_T$ exceeding a certain threshold. The trigger efficiency of each seed/trigger is calculated with the _tag-and-probe_ method as follows.

The tag and the probe are randomly assigned to the leading and sub-leading jet of the event. The leading and sub-leading scouting jets are matched to two offline jets by requiring an angular distance $\Delta R \leq 0.2$. 

The scouting tag is required to have a $p_T$ above the trigger threshold. The efficiency is then defined as the ratio of events where a scouting probe is above the trigger threshold to the total number of events passing the tag requirement.

The efficiency is calculated as a function of the average offline jet $p_T$ defined as:

$p_T^{average} = \frac{p_{T,1} + p_{T,2}}{2},$

where $p_{T,1}$ and $p_{T,2}$ refer to the $p_T$ of the two leading offline jets.

This is all executed by the Coffea 0.7 processor [processors/trigger.py](processors/trigger.py). The output is as histogram.

### Submission

To create the histogram using DASK, submit the jobs with  [submit_dask_lpc_trigger.py](submit_dask_lpc_trigger.py).

### Plotting

To create plots to evaluate the trigger efficiency, use the following notebook: [trigger/trigger.ipynb](trigger/trigger.ipynb). 

In this notebook, a trigger threshold, denoted $p_T^{average, 0.95}$, is computed for each trigger by fitting the data points of the trigger efficiency curve with

$f(p_T^{average}, a_1, a_2) = \frac{1}{2}\left( 1 + \text{erf}\left(\frac{p_T^{average}-a_1}{\sqrt{2} a_1} \right)\right)$,

where erf() refers to the error function while $a_1$ and $a_2$ are parameters of the function.

The value of $p_T^{average, 0.95}$ for each trigger is selected by finding the transverse momentum that results in $f(p_T^{average, 0.95}) = 0.95$. Each bin corresponds to a different trigger selection, with $p_T^{average, 0.95}$ equal to the bin's minimum value. To avoid any bias in the event selection, each bin is filled only by the events selected by the corresponding trigger (events selected by other triggers are ignored). For example, the bin ranging from 500 GeV to 550 GeV is filled by events selected by `HLT_PFJet450` whose $p_T^{average, 0.95}$ is approximately 500 GeV.

# JES

The _tag-and-probe_ technique is used, in which the leading scouting jet is chosen as probe and the sub-leading jet as tag. The scouting tag and probe are paired with two offline jets by requiring that the angular distance $\Delta R \leq 0.2$. The offline jets paired with the scouting probe and tag are referred to as offline probe and tag, respectively.

The JES is defined as:

$\text{JES} = \frac{\langle  p_T^{\text{scouting}} \rangle}{\langle  p_T^{\text{offline}} \rangle}$

and in order to derive it, the following four quantities are needed:

1. $\langle \frac{p_T^{\text{scouting,probe}}}{p_T^{\text{offline,tag}}} \rangle \text{ as a function of } p_T^{\text{offline,tag}},$

2. $\langle \frac{p_T^{\text{offline,probe}}}{p_T^{\text{offline,tag}}} \rangle \text{ as a function of } p_T^{\text{offline,tag}},$

3. $\langle p_T^{\text{scouting,probe}} \rangle \text{ as a function of } p_T^{\text{offline,tag}},$

4. $\langle \frac{p_T^{\text{scouting,probe}}}{p_T^{\text{offline,probe}}} \rangle \text{ as a function of } p_T^{\text{offline,tag}}$.

These quantities are computed as histograms using the Coffea 0.7 processor [processors/jes.py](processors/jes.py).

### Submission

To create the histograms using DASK, submit the jobs with  [submit_dask_lpc_jes.py](submit_dask_lpc_jes.py).

### Plotting

The JES is then derived by following the three steps outlined below, as implemented in the following notebook: [jes/jes.ipynb](jes/jes.ipynb).

- Eq. 1 is divided by Eq. 2. The division results our definition of JES as a function of $p_T^{\text{offline,tag}}$.
- Eq. 3 is then used to map the result of Step 1 from $p_T^{\text{offline,tag}}$ to $\langle p_T^{\text{scouting,probe}} \rangle$.
- Finally, the standard deviation of Eq. 4 is used to assign the uncertainty on the result of Step 2.

# JER

The JER of scouting and offline reconstructed jets are derived and compared by calculating their ratio. The measurement is performed using the dijet asymmetry technique, which exploits the dijet $p_T$-balancing method.

#### Asymmetry

The asymmetry of a dijet event is defined as

$A = \frac{p_{T,1} - p_{T,2}}{p_{T,1} + p_{T,2}},$

where $p_{T,1}$ and $p_{T,2}$ refer to the randomly ordered transverse momenta of the two leading jets.

The asymmetry is expected to be close to 0, but may vary due to measurement biases and errors in reconstruction. The distribution of asymmetries mimics a Gaussian distribution with long tails.

In the ideal case where the two jets of the dijet event are located in the same $\eta$ region and have perfectly balanced momenta, the JER can be computed from the width of the asymmetry distribution according to 

$\frac{\sigma(p_T)}{p_T} = \sigma(A) \times \sqrt{2}.$

Here, $\sigma(A)$ is computed as the _effective resolution_ and is achieved by finding the smallest interval containing 68% ($\pm 1$%) of the events, and dividing that interval by 2.

#### Simulated jet response

In order to account for the missing calibration of the JES, it is necessary to account for the simulated jet response ($\mathcal{R}^\text{simulation}$) when measuring the JER. This is a common practice when estimating the JER to avoid a bias due to an imperfect JES calibration.

The $\mathcal{R}^\text{simulation}$ is accounted for in the calculation of JER as

$\frac{\sigma(p_T)}{p_T} = \frac{ \sigma(A) \times \sqrt{2} }{\langle \mathcal{R}^\text{simulation} \rangle}.$

#### $\alpha$ extrapolation

To account for the radiation imbalance bias, the measurement of the JER is performed four times with decreasing amounts of extra jet activity. The JER is then extracted by extrapolating the extra activity to zero. The variable $\alpha$ represents the extra jet activity, and the four inclusive $\alpha$ bins used are $\alpha < 0.2$, $\alpha < 0.15$, $\alpha< 0.1$, and $\alpha < 0.05$. 

Here, $\alpha$ is defined as

$\alpha = \frac{p_T^{average}}{p_{T,3}},$

where $p_{T,3}$ is the $p_T$ of the third leading jet. The equation equals zero if the event contains exactly two jets.

#### Summary

The above steps are computed using the Coffea 0.7 processor [procesors/response.py](processors/response.py) [processors/jer.py](processors/jer.py). The output are histograms.

### Submission

To create the response histograms using DASK, submit the jobs with [submit_dask_lpc_response.py](submit_dask_lpc_response.py).

To create the asymmetry histogram using DASK, submit the jobs with [submit_dask_lpc_jes.py](submit_dask_lpc_jes.py).

### Plotting

Finally, to create plots of the JER, run:

```
python jer/plot_jer.py
```
