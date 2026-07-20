# Abstract Draft

**Project Title:** Hierarchical Bayesian learning of prior confidence during perceptual estimation.

**Team Members:** Anirban Bhattacharya, Salma elhanchi, Rachel Wise, Rommana Ruamkonthong, Valeria Kolchina

## Abstract

Human perceptual estimates are often described by Bayesian models that integrate prior expectations with uncertain sensory evidence. However, in a motion-direction estimation task, participants produced response distributions with peaks near both the prior mean and the presented motion direction. This bimodality motivated a Switching observer that selects between prior-based and sensory-based estimates rather than combining them into a single posterior. The present project asks whether the same behavior can instead be explained by a hierarchical Bayesian observer that updates confidence in the prior across trials. We hypothesize that trial-by-trial learning of prior precision may reproduce both unimodal and bimodal response distributions and provide a competitive account of behavior relative to the Switching observer.

We will analyze trial-level data from 12 participants who estimated the direction of briefly presented random-dot motion. Sensory reliability was manipulated using motion coherences of 6%, 12%, and 24%, while prior reliability was manipulated across blocks using motion-direction distributions with standard deviations of 10 deg, 20 deg, 40 deg, and 80 deg. The principal variables are presented direction, reported direction, prior width, motion coherence, trial position, and the model's evolving estimate of prior confidence. The hierarchical Bayesian model will be fitted separately to each participant by maximizing the likelihood of their trial-level responses while preserving trial and block order. Model validity will be assessed through controlled simulations, inspection of prior and posterior distributions, convergence of learned prior confidence, posterior-predictive checks, and parameter-recovery analysis. We will then implement the original Switching observer and compare the two models using negative log-likelihood, AIC, BIC, sequence-preserving cross-validation, and their ability to reproduce the shape and bimodality of individual response distributions. This study will test whether switching-like perceptual behavior requires an explicit selection mechanism or can emerge from adaptive hierarchical Bayesian inference.

## Expected Results And Significance

We expect the hierarchical model to reveal substantial individual differences in prior-confidence learning and to explain some subjects through continuous integration while approximating switching-like behavior in others. This work will clarify whether bimodal perceptual estimates require an explicit switching mechanism or can emerge from adaptive hierarchical inference, providing a more unified account of how humans learn and use uncertainty during perception.

**Keywords:** hierarchical Bayesian inference; perceptual estimation; prior confidence; sensory uncertainty; switching observer
