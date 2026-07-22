Title: A hierarchical Bayesian observer that learns prior confidence trial-by-trial during perceptual estimation

Team members: [add names]

Background: Perceptual estimation is dominantly modelled as optimal Bayesian integration: observers combine sensory evidence with prior expectations into a single percept varying unimodally between the two. Laquitaine and Gardner (2018) reported a sharp divergence: human direction estimates were often bimodal — sometimes near the stimulus, sometimes the prior — and were better explained by an observer that switches between prior and evidence each trial than by one integrating them. Crucially, their switching model treats prior reliability as fixed within each block: it captures that observers weight the prior differently across conditions, but leaves unexplained how that confidence is acquired, even though priors differ in width and are learned from feedback.

Research question: Can a single integrating observer reproduce these patterns if it learns how much to trust the prior over trials? We hypothesize that an observer that always integrates, but treats the prior's width as uncertain and learned, (i) produces bimodal estimates with no switching rule, and (ii) recovers a prior width that tracks each block.

Methods: We fit the observer to direction estimates from 12 participants across four prior widths (SD 10–80 degrees) and three coherences. It carries a belief over prior concentration, updated each trial from feedback. Because the width changes between blocks, a purely Bayesian belief would never re-learn; a forgetting term leaks the belief toward its uncertain initial state each trial — a deliberate departure from strict Bayes. Parameters are fit by maximum likelihood and compared to the switching observer by AIC.

Expected results: We expect bimodality to emerge from integration alone, strongest far from the prior mean, and the inferred width to fall from wide to narrow blocks — reproduced in preliminary fits across participants.

Conclusion and significance: This would show that trial-by-trial learning of prior confidence, not strategy-switching, can account for signatures read as switching, reframing an either/or choice as graded, adaptive integration.

Keywords: hierarchical Bayesian inference; perceptual estimation; prior confidence; online learning; motion direction
