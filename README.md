# Using Markov Chain Monte Carlo (MCMC) to decipher encrypted texts

This repository shows how MCMC can be used in order to decipher encrypted texts.

In breif, the algorithm works by randomly substituting letters in a cyphertext with one another, before assessing the likelihood of text being written in English. This likelihood is assessed as the sum of log conditional probabilities of letters appearing after a given letter. I.e. t is commonly followed by o, however is infrequently followed by z. Hence 'to' is more likely to belong to the English language that 'tz'.