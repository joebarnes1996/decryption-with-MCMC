#============================================================================
"""
1. Load libraries
"""
# Import all the usual things
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import tools for reading files and processing strings
import string
from pathlib import Path 
import os

# Also get tools to read the dictionary
# of probabilities back.
import json







#============================================================================
"""
2. Load the json file for probabilities of letters
"""
# set the path to the location of the data
os.chdir(r'C:\Users\joeba\github_projects\MCMC_decipher\data')

# load the probabilitise
prob_file_obj = open('LogPairProbDict.json', 'r')
json_from_file = prob_file_obj.read()
log_pair_probs = json.loads(json_from_file)









#============================================================================
"""
2. Create relevant functions
"""

# given a dictionary of log(p(a, b)), extract alphabet of allowed characters.
# NOTE - alphabet here means allowable characters.
def extract_alphabet(probDict):
    
    # examine input to get alphabet of allowed characters
    my_alphabet = list(probDict.keys())
    my_alphabet_str = ''.join(my_alphabet)
    
    return my_alphabet_str

# test it
extract_alphabet(log_pair_probs)




# standardise a text so only decided characters are allowed
def standardise_text(raw_text, allowed_chars, replacement_char= ' '):
    
    # make all characters lowercase
    raw_text = raw_text.lower()
    
    # replace any characters that arent part of our list (allowedChars) 
    # using a replacement of 'replacementChar'
    standardised_text = ''
    
    for char in raw_text:
        
        if allowed_chars.find(char) == -1: # char isn't in the allowed list
            
            standardised_text = standardised_text + replacement_char
            
        else:
            
            standardised_text = standardised_text + char
            
    return standardised_text

# test it - replace 'unknown' characters with a *
test_text = 'Where would heavy metal be without Úmlaut?'
standardise_text(test_text, ' abcdefghijklmnopqrstuvwxyz', '*')        
            






# create a function to map each character in a string to another character,
# based on a cypher (dictionary)
def apply_cypher(msg, cypher_dict):
    
    result = ''
    
    for char in msg:
        
        result += cypher_dict[char]
        
    return result

# test it - result should be 'cbabc'
apply_cypher('abcba', {'a':'c', 'b':'b', 'c':'a'})





# find the inverse of a given cypher
def invert_cypher(cypher_dict):
    
    inverse_dict = dict.fromkeys(cypher_dict.keys())
    for plain_text_char in cypher_dict.keys():
    
        cypher_text_char = cypher_dict[plain_text_char]
        
        inverse_dict[cypher_text_char] = plain_text_char
        
    return inverse_dict

# Testing: result should be {'a': 'c', 'b': 'a', 'c': 'b'}
invert_cypher( {'a':'b', 'b':'c', 'c':'a'} )




# convert a string of characters to a cypher dict. The first character would be
# a, the second b, and so on
def cypher_str_to_dict(cypher_str):
    
    alphabet = sorted(cypher_str)
    
    cypher_dict = dict.fromkeys(alphabet)
    
    for j in range(len(alphabet)):
        
        cypher_dict[alphabet[j]] = cypher_str[j]
    
    return cypher_dict
        
# testing result should give {a:b, b:c, c:a}
cypher_str_to_dict('bca')
    


# convert a cypher to a string. I.e. a string made up of the list of dict values
def cypher_dict_to_str(cypher_dict):
    
    return ''.join(list(cypher_dict.values()))

# should retrun a test string
test_str = 'bca'
cypher_dict_to_str(cypher_str_to_dict(test_str))





# create a random substitution cypher for a given alphabet
def random_cypher(alphabet_str, seed=0):
    
    random.seed(seed)
    
    # put alphabet in standard order
    alphabet = sorted(alphabet_str)
    

    
    # generate a shuffled version of the alphabet
    scrambled_alphabet = alphabet.copy()
    random.shuffle(scrambled_alphabet)
    
    # assemble dictionary of substituions
    cypher = dict.fromkeys(alphabet, '')
    
    for j in range(len(alphabet)):
        
        cypher[alphabet[j]] = scrambled_alphabet[j]
        
    return cypher

# do a small test
small_alphabet = 'abcdefg'
random_cypher(small_alphabet)






#============================================================================
"""
3. Create the MCMC functions
"""



#============================================================================
"""
3.1 Create a function to evaluate log-likelihood of the converted text
    belonging to the english language.
    

NOTE - The log-likelihood of it being the english is taken 
       as the sum of log probabilities of each character 
       following from the last.
       
       These probabilities are evaluated from analysing the book of 
       'Dream of the Red Chamber', which finds the frequency for which all
       characters follow any other character, which are then stored in the
       log_pair_probs dictionary.

"""

def log_likelihood(msg, log_pair_probs=log_pair_probs):
    
    log_like = 0
    
    for i in range(1, len(msg)):
        
        # model as a = t_{n} and b = t_{n+1}
        
        a = msg[i-1]
        b = msg[i]
        
        # add log( p ( b, a)), the log of the probability b will follow a
        log_like += log_pair_probs[a][b]

    return log_like


# compare a 'likely' phrase against an 'unlikely' phrase
phrase_english =  'a dog went for a walk'
phrase_nonsense = 'dskjgow8ey8c34t4hf389'


test = log_likelihood(phrase_english)
test_2 = log_likelihood(phrase_nonsense)

# if log likelihood is higher (closer to being +ve), then it is more probable
# of being real

if test > test_2:
    
    print(phrase_english, 'is more likely than', phrase_nonsense)

else:
    
    print(phrase_nonsense, 'is more likely than', phrase_english)





#============================================================================
"""
3.2 Decypher using MCMC
"""

def decipher_with_MCMC(cypher_text, log_pair_probs, n_samples, burn_in, seed):
    
    # set random seed
    np.random.seed(0)
    
    # Examine the input to get the alphabet of allowed characters
    my_alphabet = extract_alphabet(log_pair_probs)
    
    # Step 1 - initialise MCMC run by choosing a decryption key at random. 
    #          This is equivalent to sampling from a uniform prior.
    crnt_cypher_dict = random_cypher(my_alphabet, seed=seed)
    
    
    # Step 2 - Decrypt the cyphertext using crntCypherDict
    crnt_decrypt_dict = invert_cypher(crnt_cypher_dict)
    crnt_plain_text = apply_cypher(cypher_text, crnt_decrypt_dict)
    
    # Step 3 - Compute the current loglikelihood
    crnt_log_like = log_likelihood(crnt_plain_text, log_pair_probs)
    
    
    # Do the sampling
    n_proposed = 0
    n_accepted = 0
    sample_num = 0
    samples   = [''] * n_samples # initially empty
    
    
    Ss = []
    
    # accepted proposals
    accepted = []
    
    while sample_num < n_samples:
        
        # step 4 - generate a proposal
        #          (choose a pair of symbols from the alphabet and make a new
        #           cypher that swaps characters assigned to the pair).
        
        # randomly pick 2 characters from myAlphabet to make a random proposal
        sample_pair = random.sample(my_alphabet, 2)
        
        swap_1 = sample_pair[0]
        swap_2 = sample_pair[1]        
        
        # get the values of each swap
        swap_1_val = crnt_cypher_dict[swap_1]
        swap_2_val = crnt_cypher_dict[swap_2]
        
        # make a proposal dictionary
        proposal_cypher_dict = crnt_cypher_dict.copy()
        
        # change the values of the pairs
        proposal_cypher_dict[swap_1] = swap_2_val
        proposal_cypher_dict[swap_2] = swap_1_val
        
        proposal_decrypt_dict = invert_cypher(proposal_cypher_dict)

        
        
        # step 5 - create the proposed plaintext
        proposal_plain_text = apply_cypher(cypher_text, proposal_decrypt_dict)
        
        
        # step 6 - calcluate the proposed log-likelihood, and update proposals
        proposal_log_like = log_likelihood(proposal_plain_text, log_pair_probs)
        n_proposed += 1
        
        # Step 7 - M-H acceptance rule. 
        # create the ratio (convert back from log-likelihood)
        P = np.exp(proposal_log_like - crnt_log_like)
        
        # accept with a probability of P
        if P > np.random.uniform(0,1):
            
            
            crnt_cypher_dict = proposal_cypher_dict
            crnt_log_like = proposal_log_like 
            n_accepted += 1
            
            accepted.append([n_proposed, n_accepted])
            

        
        # only record the number of samples after the 
        # burn in period
        if n_proposed >= burn_in:
            
            samples[sample_num] = cypher_dict_to_str(crnt_cypher_dict)
            sample_num += 1

    
        Ss.append(cypher_dict_to_str(crnt_cypher_dict))
    
    # print the acceptance ratio
    print(n_accepted / n_proposed)
    return samples, Ss, accepted
            
        
        









#============================================================================
"""
5. Apply it to cypher text
"""

# set the path to load the cypher text
cypher_text = Path('cyphertext.txt').read_text()

# show the original text
print(cypher_text)


# apply to the cyphertext
samples, Ss, acceptRatio = decipher_with_MCMC(cypher_text, log_pair_probs, 
                                              n_samples=50, burn_in=10000, 
                                              seed=2)

# use the found cypher to convert the cypher text baack to plaintext
final_decipher_dict = invert_cypher(cypher_str_to_dict(samples[-1]))
final_deciphered_text = apply_cypher(cypher_text, final_decipher_dict)


# show the decrypted text
print(final_deciphered_text)




