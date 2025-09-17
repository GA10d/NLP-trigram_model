"""
python -i trigram_model.py data/brown_train.txt data/brown_test.txt
Zhewen Guo(zg2567)
"""
import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2025 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    ngrams = []
    # Make Sure n >= 1
    if n < 1:
        raise ValueError("This should work for arbitrary values of n >= 1")
    
    # Pad the sequence by inserting n-1 START and 1 STOP
    processed_sequence = (n-1) * ["START"] + sequence +["STOP"]
    # print ("precessed_sequence:\n",processed_sequence)
    for i in range(len(sequence)+1):
        ngram = tuple(processed_sequence[i:i+n])
        ngrams.append(ngram)
    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
        print("Training on corpus:", corpusfile)
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Template is changed here！！！:
        # I add total count here to calculate unigram probability later
        # And the figure would update in count_ngrams method
        self.total_count = 0
        # Store corpusfile, for use in perplexity calculation
        self.corpusfile = corpusfile

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)




    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            # print("unigrams:\n",unigrams)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)
            # Count unigrams
            for unigram in unigrams:
                self.total_count += 1
                if unigram in self.unigramcounts:
                    self.unigramcounts[unigram] += 1
                else:
                    self.unigramcounts[unigram] = 1
            # Count bigrams
            for bigram in bigrams:
                if bigram in self.bigramcounts:
                    self.bigramcounts[bigram] += 1
                else:
                    self.bigramcounts[bigram] = 1
            # Count trigrams
            for trigram in trigrams:
                if trigram in self.trigramcounts:
                    self.trigramcounts[trigram] += 1
                else:
                    self.trigramcounts[trigram] = 1

            # Test print 
            # print(next(iter(tm.unigramcounts.items())))
            # print(next(iter(tm.bigramcounts.items())))
            # print(next(iter(tm.trigramcounts.items())))
        
        return 

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # P(v|u,w) = P(u,w,v) / P(w,u)
        # trigram = (u,w,v)

        u,w,v = trigram
        bigram = (u, w)

        trigram_count = self.trigramcounts.get(trigram, 0)
        bigram_count = self.bigramcounts.get(bigram, 0)
        unigram_count = self.unigramcounts.get((v,), 0)


        # If trigram_count == 0 then bigram_count must be 0 too
        # However there is one situation that trigram_count > 0 but bigram_count == 0
        # e.g. trigram_count(('START', 'START', 'the')) while there is no bigram_count(('START', 'START'))
        if bigram_count > 0:
            # print("bigram_count > 0: ", trigram_count)
            return trigram_count / bigram_count
        # Use P(v|u,w) = P(v) to deal with zero counts
        else:
            # print("bigram_count <= 0: ", unigram_count, self.total_count)
            return unigram_count/ self.total_count

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # P(v|w) = P(w,v) / P(w)
        # bigram = (w,v)
        (w,v) = bigram
        bigram_count = self.bigramcounts.get(bigram, 0)
        w_unigram_count = self.unigramcounts.get((w,), 0)
        v_unigram_count = self.unigramcounts.get((v,), 0)

        if w_unigram_count > 0:
            # print("w_unigram_count > 0: ", bigram_count, w_unigram_count)
            return bigram_count / w_unigram_count
        else:
            # print("w_unigram_count <= 0: ", v_unigram_count, self.total_count)
            return v_unigram_count / self.total_count
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts.get(unigram, 0) / self.total_count

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # Calculate the next word based on the previous two words
        # Judge if the next word is STOP
        # Judge if the length of the sentence reaches t
        # Loop

        sentence = []
        for i in range(t):
            if i == 0:
                bigram = ("START", "START")
            elif i == 1:
                bigram = ("START", sentence[0])
            else:
                bigram = (sentence[i-2], sentence[i-1])

            filtered = {k: v for k, v in self.trigramcounts.items() if k[:2] == bigram}
            # max_key = max(filtered, key=lambda k: filtered[k])

            # Randomly choose next word based on keys' frequency
            next_word = random.choices(list(filtered.keys()), weights=list(filtered.values()))[0]
            sentence.append(next_word[2])
            if next_word[2] == "STOP":
                break
        # print("Generated Sentence: ", " ".join(sentence))
        return sentence       

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        u,w,v = trigram
        bigram = (w,v)
        unigram = (v,)

        return lambda1*self.raw_trigram_probability(trigram) + \
               lambda2*self.raw_bigram_probability(bigram) + \
               lambda3*self.raw_unigram_probability(unigram)
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        log_prob = 0.0
        # Not sure if sentence is a list or a string
        if isinstance(sentence, list):
            sequence = sentence
        else:
            sequence = sentence.lower().strip().split()

        trigrams = get_ngrams(sequence, 3)
        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)
            if prob > 0:
                log_prob += math.log2(prob)
            else:
                return float("-inf")
        return log_prob 


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_log_prob = 0.0
        total_tokens = 0

        # Not sure if corpus is a file path or an iterator
        if isinstance(corpus, str):
            # print("corpus is a string, treat it as a file path")
            generator = corpus_reader(corpus, self.lexicon)
        else:
            # print("corpus is an iterator")
            generator = corpus
        
        for sequence in generator:
            # print(sequence)
            total_log_prob += self.sentence_logprob(sequence)
            unigrams = get_ngrams(sequence, 1)
            for unigram in unigrams:
                total_tokens += 1
        l = total_log_prob/total_tokens
        return 2**(-l)



def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        # testdir1 and training_file1 is high level essays
        # testdir2 and training_file2 is low level essays
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            
            total += 1
            if pp1 < pp2:
                correct += 1
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            if pp2 < pp1:
                correct += 1
                                                               
        return correct*1.0/total

if __name__ == "__main__":

    # $ python -i trigram_model.py
    # data_file = os.path.join("data","brown_train.txt")
    # test_file = os.path.join("data","brown_test.txt")
    # model = TrigramModel(data_file) 
    model = TrigramModel(sys.argv[1]) 
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py data/brown_train.txt data/brown_test.txt
    # >>> 
    print("Part 1 - Extracting n-grams from a Sentence")
    print(get_ngrams(["Guo","Zhe","Wen"], 1))
    print(get_ngrams(["Guo","Zhe","Wen"], 2))
    print(get_ngrams(["Guo","Zhe","Wen"], 3))

    print("\nPart 2 - Counting n-grams in a Corpus")
    print("trigramcounts: ", model.trigramcounts[('START','START','the')])
    print("bigramcounts: ", model.bigramcounts[('START','the')])
    print("unigramcounts: ", model.unigramcounts[('the',)])
    
    print("\nInterlude - Generating text")
    for i in range(3):
        print(f"senctence{i+1}: ", model.generate_sentence())
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    print("\nPart 6 - Perplexity")
    # sys.argv[1] is the training corpus
    # sys.argv[2] is the testing corpus
    print("Perplexity of the training corpus: ", model.perplexity(sys.argv[1]))
    print("Perplexity of the testing corpus: ", model.perplexity(sys.argv[2]))

    # Essay scoring experiment: 
    print("\nPart 7 - Using the Model for Text Classification")
    acc = essay_scoring_experiment('data/ets_toefl_data/train_high.txt', 'data/ets_toefl_data/train_low.txt', "data/ets_toefl_data/test_high", "data/ets_toefl_data/test_low")
    print("Accuracy of essay scoring experiment: ", acc)

