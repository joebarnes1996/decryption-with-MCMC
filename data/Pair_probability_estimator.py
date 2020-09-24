"""" 
1. Import all the usual things
"""
import math
import random
import numpy as np
import os

# Also stuff for reading files processing strings
import string
import collections
from pathlib import Path 

# And, finally, stuff to write dictionaries out to files
# and read them back again.
import json



# change working directory
os.chdir(r'C:\Users\joeba\github_projects\MCMC_decipher\data')





"""
2. Standardise the text
"""
def standardiseText( rawText, allowedChars = ' abcdefghijklmnopqrstuvwxyz',
                    replacementChar=' ' ):
    # Make all the characters lower case
    rawText = rawText.lower()

    # Replace any characters that aren't part of our list
    # of allowed characters with the replacement character
    standardisedText = ""
    for char in rawText:
        if allowedChars.find(char) == -1:
            # char isn't one of the allowed ones
            standardisedText = standardisedText + replacementChar
        else:
            standardisedText = standardisedText + char
            
    return( standardisedText )

# Do a small test
testText = "Where would heavy metal be without the ümlaut?"
standardiseText(testText, replacementChar='*' )








"""
3. 'Slurp' the training text, count pairs and estimate p(a,b)
"""
def findLogPairProbs( trainingTextFiles, allowedChars ):
    ############################################
    # Read and standardise the training texts
    ############################################
    # Read the whole training text into a single string 
    # See https://stackoverflow.com/questions/1631897/python-file-slurp
    rawTrainingText = ''
    for textFile in trainingTextFiles:
        rawTrainingText += Path(textFile).read_text()
        
    print( 'The training text is ' + str(len(rawTrainingText)) + ' characters long.')
    
    ###########################################
    # Standardise the training text
    ###########################################
    # Make sure the list of allowed characters doesn't contain any repeats
    charSet = sorted( allowedChars ) 
    charSetStr = ''.join(charSet)
    nSymbols = len(charSetStr)
    
    # Standardise the text
    stdTrainingText = standardiseText( rawTrainingText, charSetStr, ' ' )        
    
    ##########################################
    # Count appearances of pairs
    ##########################################
    # Build a dictionary whose keys are the allowed characters
    # and whose values are integers. They'll eventually be counts.
    emptyCountDict = dict.fromkeys( charSet, 0 )

    # Build a dictionary whose keys are the allowed characters 
    # and whose values are copies of the empty count dictionary
    pairCounts = dict.fromkeys( charSet )
    for char in pairCounts.keys():
        pairCounts[char] = emptyCountDict.copy()

    # Now count appearances of pairs
    for j in range(1, len(stdTrainingText)):
        firstChar = stdTrainingText[j-1]
        secondChar = stdTrainingText[j]
        pairCounts[firstChar][secondChar] += 1
        
    ###########################################
    # Estimate the probabilities, or rather, 
    # their logs
    ###########################################
    # Build a dictionary-of-dictionaries that holds
    # the parameters of the Dirichlet posteriors.
    priorAlpha = 2 # favours broadly uniform, nonzero probabilities for letter pairs

    dirichletPosterior = dict.fromkeys( charSet )
    for firstChar in dirichletPosterior.keys():
        dirichletPosterior[firstChar] = dict.fromkeys( charSet )
        for secondChar in dirichletPosterior[firstChar].keys():
            dirichletPosterior[firstChar][secondChar] = pairCounts[firstChar][secondChar] + priorAlpha
            
    # Get the MAP estimates for the probabilities
    logPairProbs = dict.fromkeys( charSet )
    for firstChar in logPairProbs.keys():
        # Add up all the entries in the posterior for this row
        posteriorAlphaSum = 0.0
        for secondChar in dirichletPosterior[firstChar].keys():
            posteriorAlphaSum += dirichletPosterior[firstChar][secondChar]

        # Initialise the result
        logPairProbs[firstChar] = dict.fromkeys( charSet, 0.0 )

        # Get the logs of the maximum-a-posteriori estimates por the probs
        logNomalisation = math.log( posteriorAlphaSum - len(dirichletPosterior[firstChar].keys()) )
        for secondChar in logPairProbs[firstChar].keys():
            logPairProbs[firstChar][secondChar] = math.log(dirichletPosterior[firstChar][secondChar] - 1) - logNomalisation
    
    return( logPairProbs )





"""
4. Applying tools and estimating p(a,b)
"""

trainingTextFiles = ['CaoJoly_DreamOfTheRedChamber_Vol1.txt',
                     'CaoJoly_DreamOfTheRedChamber_Vol2.txt']

commonChars = 'abcdefghijklmnopqrstuvwxyz0123456789 ,.?!:;'

logPairProbs = findLogPairProbs(trainingTextFiles, commonChars)

# sums across rows

nSymbols = len(logPairProbs.keys())
probMat = np.zeros( (nSymbols, nSymbols) )

row = 0
for firstChar in logPairProbs.keys():
    probMat[row,:] = np.exp( np.array(list(logPairProbs[firstChar].values())))
    row += 1
    

np.sum( probMat, axis=1)


"""
5. Write probabilities to a file
"""
# Write the dictionary out to a file
jsonForProbDict = json.dumps(logPairProbs)

myFileObj = open("LogPairProbDict.json", "w")

myFileObj.write(jsonForProbDict)

myFileObj.close()





"""
6. Read it back
"""

# Read it back 
myFileObj = open("LogPairProbDict.json", "r")
jsonFromFile = myFileObj.read()
dictFromFile = json.loads(jsonFromFile)

# Check whether the two versions agree
dictFromFile == logPairProbs










