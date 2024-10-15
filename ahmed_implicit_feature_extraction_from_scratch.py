import Stemming
import nltk
import math
import numpy as np, numpy.random
from nltk.corpus import wordnet as wn
import NDCG
from collections import defaultdict
import copy

Data_Directory='./Annotated Data/iPod/'
DataFile='iPod.final'
Results_Directory='./Results/iPod/EM_Sentence/'
EquivalenceFile='Selected Features.txt'

class ReviewPreprocessor:


    def equivalent_aspects_from_file(self, filePath):

        equivalentAspects = {}
        with open(filePath, 'r') as inputfile:
            for line in inputfile:
                aspects = line.strip().split(',')
                for aspect in aspects:
                    equivalentAspects[aspect] = aspects[0]

        return equivalentAspects

    def stopword_from_file(self, filePath):

        stopword = [line.rstrip('\n') for line in open(filePath)]
        return stopword



    def review_lines_from_file(self, filePath):

        numReview = 0
        reviewLines = []
        with open(filePath, 'r') as inputfile:
            for line in inputfile:
                if line.startswith('[t]') or line.startswith('*'):
                    numReview += 1
                reviewLines.append(line)

        return numReview, reviewLines


    def train_data_From_reviews(self, reviewLines):

        trainingData = [Stemming.perform_Stemming(line) for line in reviewLines]
        return trainingData






class aspectSentiment:

  def __init__(self, aspect, sentiment, implicit):
    self.aspect = aspect
    self.sentiment = sentiment
    self.implicit=implicit


class Stat:

    def __init__(self,  stopword, punctuation, equivalent_aspects, trainData):
        self.numWord = 0
        self.numRvw = 0
        self.numLine = 0
        self.trainData = trainData
        self.stopword = stopword
        self.punctuation = punctuation
        self.raw_line_by_rvw = []  # RawReviews[]
        self.word_frequency_ByRvwIdxSentenceIdx = []  # Reviews
        self.groundtruth_ByRvwIdxSencenteIdx = []  # GroundTruth=[]
        self.IDF = defaultdict(float)
        self.word_frequency = defaultdict(int) #Collection
        self.EquivalentAspects = equivalent_aspects
        self.ExplicitFeatures = defaultdict(int)
        self.TopicModel = defaultdict(lambda: defaultdict(float))
        self.background_prob = None



    def remove_punctuation(self, input_string):
        input_string1 = ''
        for word in input_string.split():
            if word not in self.stopword:
                input_string1 = input_string1 + ' ' + word
        for item in self.punctuation:
            input_string1 = input_string1.replace(item, ' ')

        return input_string1


    def is_new_review(self, line):
        return line.startswith('[t]') or line.startswith('*')


    def extend_container_for_new_review(self):
        self.word_frequency_ByRvwIdxSentenceIdx.append(list())
        self.raw_line_by_rvw.append(list())
        self.groundtruth_ByRvwIdxSencenteIdx.append(list())
        #self.groundTruthByRvwIdxSencenteIdx[-1].append(list())


    def separate_rvw_sentence_from_aspect(self, rvw_with_aspect):
        aspect, review_Sentence = rvw_with_aspect.replace('(', '').replace(')', '').replace('-', '').split('##')
        review_Sentence = self.remove_punctuation(review_Sentence)
        return aspect, review_Sentence.strip()


    def process_word_in_sentence(self, review_sentence):
        sentence_word_frequency = defaultdict(int)
        for word in review_sentence.split():
            sentence_word_frequency[word] += 1
            self.word_frequency[word] +=1

        for word in sentence_word_frequency:
            #self.word_frequency_ByRvwIdxSentenceIdx[][][word] += sentence_word_frequency[word]
            self.IDF[word] +=1

        return sentence_word_frequency

    def remove_punctuation(self,input_string):
        input_string1 = ''
        for word in input_string.split():
            if word not in self.stopword:
                input_string1 = input_string1 + ' ' + word
        for item in punctuation:
            input_string1 = input_string1.replace(item, ' ')
        return input_string1




    def normalize_frequency(self, frequency_dict):

        total_occurance = sum(frequency_dict.values())
        for key in frequency_dict:
            frequency_dict[key] /= total_occurance
        return frequency_dict

    def calc_background_prob(self, collection, numWord):
        backgraound_prob = {}

        for word in collection:
            backgraound_prob[word] = collection[word] / numWord
        return backgraound_prob

    def calc_topicmodel(self, TopicModel, IDF , word_frequency, numLines):
        for word in word_frequency:
            for aspect in TopicModel:
                if word not in TopicModel[aspect]:
                    TopicModel[aspect][word] = 0

        for aspect in TopicModel:
            for word in TopicModel[aspect]:
                TopicModel[aspect][word] = math.log(1 + TopicModel[aspect][word]) * math.log(
                    1 + (numLines / IDF[word]))

        for aspect in TopicModel:
            sumOfTFIDF = sum(TopicModel[aspect].values())
            for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get,
                               reverse=True):  # why need to sort !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                TopicModel[aspect][word] = (TopicModel[aspect][word] + 1) / (sumOfTFIDF + len(word_frequency))

        return TopicModel

    def add_synonym_to_topicmodel(self):
        for aspect in self.TopicModel:
            for word in sorted(self.TopicModel[aspect], key= self.TopicModel[aspect].get, reverse=True)[:3]:
                SynonymSet = []
                syns = wn.synsets(word)
                for s in syns:
                    for l in s.lemma_names():
                        SynonymSet.append(l)
                for s in SynonymSet:
                    if s not in self.TopicModel[aspect]:
                        self.TopicModel[aspect][s] = self.TopicModel[aspect][word]

    def save_topicmodel(self, results_dir):
        with open(results_dir + 'TopicModel.txt', 'w') as outputFile:
            for aspect in self.TopicModel:
                for word in sorted(self.TopicModel[aspect], key=self.TopicModel[aspect].get, reverse=True)[:10]:
                    outputFile.write(aspect + ', ' + word + ', ' + str(self.TopicModel[aspect][word]) + '\n')
                outputFile.write('\n\n\n')

    def save_normalized_aspect_frequency(self, result_dir):
        with open(result_dir + 'ExplicitFeatures.txt', 'w') as outputfile:
            for aspect in sorted(self.ExplicitFeatures, key=self.ExplicitFeatures.get, reverse=True):
                outputfile.write(aspect + ' : ' + str(self.ExplicitFeatures[aspect]) + '\n')

    def save_model(self,result_dir):
        self.save_topicmodel(result_dir)
        self.save_normalized_aspect_frequency(result_dir)

    def randomInitPI(self, PI):
        for reviewNum in range(0, len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineNum in range(0, len(self.word_frequency_ByRvwIdxSentenceIdx[reviewNum])):
                RandomProbabilities = [np.random.dirichlet(np.ones(len(self.TopicModel)), size=1)[0]]
                idx = 0
                for aspect in self.TopicModel:
                    PI[reviewNum][lineNum][aspect] = RandomProbabilities[0][idx]
                    idx += 1
        return PI

    def expectation_step(self, lambdaB):
        for rvwIdx in range( len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineIdx in range(len(self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx])):
                for word in self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx]:
                    mysum = 0
                    for aspect in self.TopicModel:
                        mysum += self.PI[rvwIdx][lineIdx][aspect] * self.TopicModel[aspect][word]
                    for aspect in self.TopicModel:
                        self.HP[rvwIdx][lineIdx][word][aspect] = self.PI[rvwIdx][lineIdx][aspect] * self.TopicModel[aspect][
                            word] / mysum

                    self.HPB[rvwIdx][lineIdx][word] = (self.lambdaB * self.background_prob[word]) / (
                            lambdaB * self.background_prob[word] + ((1 - lambdaB) * mysum))

    def maximization_step(self):

        for rvwIdx in range(len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineIdx in range(len(self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx])):
                denom = 0
                for aspect in self.TopicModel:
                    for word in self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx]:
                        denom += self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx][word] * (1 - self.HPB[rvwIdx][lineIdx][word]) * \
                                 self.HP[rvwIdx][lineIdx][word][aspect]

                for aspect in self.TopicModel:
                    nom = 0
                    for word in self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx]:
                        nom += self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx][word] * (1 - self.HPB[rvwIdx][lineIdx][word]) * \
                               self.HP[rvwIdx][lineIdx][word][aspect]
                    try:
                        self.PI[rvwIdx][lineIdx][aspect] = nom / denom
                    except:
                        print(rvwIdx, lineIdx, aspect, nom, denom)

    def delta_param(self, previous_PI, PI):

        delta = 0.0
        for reviewNum in range(len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineNum in range(len(self.word_frequency_ByRvwIdxSentenceIdx[reviewNum])):
                for aspect in self.TopicModel:
                    delta = delta + math.pow(PI[reviewNum][lineNum][aspect] - previous_PI[reviewNum][lineNum][aspect], 2)
        return delta

    def build_model(self):
        self.process_lines(self.trainData, self.EquivalentAspects)
        self.ExplicitFeatures = self.normalize_frequency(self.ExplicitFeatures)
        self.background_prob = self.calc_background_prob(self.word_frequency, self.numWord)
        self.TopicModel = self.calc_topicmodel(self.TopicModel, self.IDF, self.word_frequency, self.numLine)
        #  self.save_model("./Results/iPod/EM_Sentence/")
        self.add_synonym_to_topicmodel()

    def define_model_param(self):
        self.HP = [
            [defaultdict(lambda: defaultdict(float)) for j in range(len(self.word_frequency_ByRvwIdxSentenceIdx[i]))]
            for
            i in range(len(self.word_frequency_ByRvwIdxSentenceIdx))]
        self.HPB = [[defaultdict(float) for j in range(len(self.word_frequency_ByRvwIdxSentenceIdx[i]))] for i in
                    range(len(self.word_frequency_ByRvwIdxSentenceIdx))]
        self.PI = [[defaultdict(float) for j in range(len(self.word_frequency_ByRvwIdxSentenceIdx[i]))] for i in
                   range(len(self.word_frequency_ByRvwIdxSentenceIdx))]

    def initialize_model_param(self):
        self.lambdaB = 0.7
        self.PI = self.randomInitPI(self.PI)

    def learn_model_param(self,mx_iter, lambdaB, delta_threshold):

        print("Starting EM------------------------")

        for i in range(mx_iter):
            print('iteration: ', i)

            self.expectation_step(lambdaB)
            previousPI = copy.deepcopy(self.PI)
            self.maximization_step()
            delta_param = self.delta_param(previousPI,self.PI)
            print('Change in param:', delta_param)
            if delta_param < delta_threshold:
                break


    def main(self, mx_iter, lamdaB, threshold):
        self.build_model()
        print ("model build completed")
        self.define_model_param()
        self.initialize_model_param()
        self.learn_model_param(max_iter, lambdaB, threshold)


equ_file_path = Data_Directory + EquivalenceFile
stopword_dir = Data_Directory+'lemur.stopwords'
data_dir = Data_Directory + DataFile

rp = ReviewPreprocessor()
stopword = rp.stopword_from_file(stopword_dir)
punctuation = "'.,:;!?1234567890"
equ_aspect = rp.equivalent_aspects_from_file(equ_file_path)
n,r = rp.review_lines_from_file(data_dir)
trainData = rp.train_data_From_reviews(r)
stat = Stat(stopword, punctuation, equ_aspect,trainData)
max_iter = 50
lambdaB = 0.7
dist_threshold = 1e-6
stat.main(max_iter,lambdaB,dist_threshold)

#print(r)
#print(td)