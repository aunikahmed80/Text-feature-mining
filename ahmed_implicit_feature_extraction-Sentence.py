import Stemming
import math
import numpy as np, numpy.random
from nltk.corpus import wordnet as wn
from collections import defaultdict
import copy

from wikipedia2vec import Wikipedia2Vec
wiki2vec = Wikipedia2Vec.load('/media/ahmed/eda2376f-27e0-48b9-b8a6-92070383dbd8/data/embedding_data/enwiki_20180420_win10_300d.pkl')
from gensim.models import Word2Vec

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


class FeatureMiningSentenceLevel:

    def __init__(self,  stopword, punctuation, equivalent_aspects, trainData):
        self.numWord = 0
        self.numRvw = 0
        self.numLine = 0
        self.trainData = trainData
        self.stopword = stopword
        self.punctuation = punctuation
        self.word_frequency_ByRvwIdxSentenceIdx = []  # Reviews
        self.IDF = defaultdict(float)
        self.word_frequency = defaultdict(int) #Collection
        self.EquivalentAspects = equivalent_aspects
        self.ExplicitFeatures = defaultdict(int)
        self.TopicModel = defaultdict(lambda: defaultdict(float))
        self.background_prob = None
        self. raw_reviews = []
        self.groundtruth = []


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


    def separate_rvw_sentence_from_aspect(self, rvw_with_aspect):
        aspect, review_Sentence = rvw_with_aspect.replace('(', '').replace(')', '').replace('-', '').split('##')
        review_Sentence = self.remove_punctuation(review_Sentence)
        return aspect, review_Sentence.strip()


    def word_freq_in_sentence(self, review_sentence):
        sentence_word_frequency = defaultdict(int)
        for word in review_sentence.split():
            sentence_word_frequency[word] += 1
        return sentence_word_frequency


    def remove_punctuation(self,input_string):
        input_string1 = ''
        for word in input_string.split():
            if word not in self.stopword:
                input_string1 = input_string1 + ' ' + word
        for item in punctuation:
            input_string1 = input_string1.replace(item, ' ')
        return input_string1


    def process_lines(self,TrainingData,EquivalentAspects):

        ExplicitFeatures = defaultdict(int)
        TopicModel = defaultdict(lambda: defaultdict(float))
        word_frequency_ByRvwIdxSentenceIdx = []
        word_frequency = defaultdict(int)
        IDF = defaultdict(int)
        raw_reviews = []
        GroundTruth = []
        numWord = 0
        numLine = 0

        for line in TrainingData:
            #i += 1  # why consider tag sentence in corpus!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            numWord += len(line.split())
            if self.is_new_review(line):
                word_frequency_ByRvwIdxSentenceIdx.append(list())
                raw_reviews.append(list())
                GroundTruth.append(list())
            else:

                review_with_aspect = line.replace('(', '').replace(')', '').replace('-', '').split('##')
                review_Sentence = self.remove_punctuation(review_with_aspect[1])
                if review_Sentence.strip() == '':
                    continue
                numLine += 1
                raw_reviews[-1].append(line)
                GroundTruth[-1].append(list())
                word_freq_in_rvw_sent = self.word_freq_in_sentence(review_Sentence)
                word_frequency_ByRvwIdxSentenceIdx[-1].append(word_freq_in_rvw_sent)
                for word in word_freq_in_rvw_sent:
                    word_frequency[word] += word_freq_in_rvw_sent[word]
                    IDF[word] += 1

                if len(review_with_aspect[0]) > 0:
                    explicitFeatures = review_with_aspect[0].split(',')
                    for feature in explicitFeatures:
                        feature_sentiment_context = feature.split('[')
                        sentiment_context = feature_sentiment_context[1].split('@')
                        if '[u]' not in feature:
                            ExplicitFeatures[EquivalentAspects[feature_sentiment_context[0].strip()]] += 1
                            for word in word_freq_in_rvw_sent:
                                TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]][word] += word_freq_in_rvw_sent[word]
                            GroundTruth[-1][-1].append(
                                aspectSentiment(EquivalentAspects[feature_sentiment_context[0].strip()],sentiment_context[0], False))
                        else:
                            GroundTruth[-1][-1].append(aspectSentiment(EquivalentAspects[feature_sentiment_context[0].strip()], sentiment_context[0], True))



        self.numLine = numLine
        self.numWord =numWord
        self.IDF = IDF
        self.word_frequency = word_frequency
        self.TopicModel = TopicModel
        self.EquivalentAspects = EquivalentAspects
        self.ExplicitFeatures = ExplicitFeatures
        self.word_frequency_ByRvwIdxSentenceIdx = word_frequency_ByRvwIdxSentenceIdx
        self.raw_reviews = raw_reviews
        self.groundtruth = GroundTruth



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
            for word in TopicModel[aspect]:#sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True):  # why need to sort !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                TopicModel[aspect][word] = (TopicModel[aspect][word] + 1) / (sumOfTFIDF + len(word_frequency))

        return TopicModel

    # def calc_topicmodel(self, TopicModel, IDF, word_frequency, numLines):
    #     model = Word2Vec.load("word2vec.model")
    #     for word in word_frequency:
    #         for aspect in TopicModel:
    #             if word not in TopicModel[aspect]:
    #                 TopicModel[aspect][word] = 0
    #
    #     for aspect in TopicModel:
    #         aspect_vec = wiki2vec.get_word_vector(aspect.lower())
    #         #aspect_vec = model.wv[aspect.lower()]
    #         for word in TopicModel[aspect]:
    #             try:
    #                 word_vec = wiki2vec.get_word_vector(word.lower())
    #                 #word_vec = model.wv[word.lower()]
    #             except KeyError as e:
    #                 print(word)
    #                 TopicModel[aspect][word] = .0005
    #                 continue
    #             cos_sim = np.dot(aspect_vec, word_vec) / (np.linalg.norm(aspect_vec) * np.linalg.norm(word_vec))
    #             TopicModel[aspect][word] = cos_sim

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
                randomProbabilities = np.random.dirichlet(np.ones(len(self.TopicModel)), size=1)[0]
                for idx, aspect in enumerate(self.TopicModel):
                    PI[reviewNum][lineNum][aspect] = randomProbabilities[idx]

        return PI


    def expectation_step(self, lambdaB):
        for rvwIdx in range( len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineIdx in range(len(self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx])):
                for word in self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx]:
                    partition_func = 0
                    for aspect in self.TopicModel:
                        partition_func += self.PI[rvwIdx][lineIdx][aspect] * self.TopicModel[aspect][word]
                    for aspect in self.TopicModel:
                        self.HP[rvwIdx][lineIdx][word][aspect] = self.PI[rvwIdx][lineIdx][aspect] * self.TopicModel[aspect][
                            word] / partition_func

                    self.HPB[rvwIdx][lineIdx][word] = (self.lambdaB * self.background_prob[word]) / (
                            lambdaB * self.background_prob[word] + ((1 - lambdaB) * partition_func))


    def maximization_step(self):

        for rvwIdx in range(len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineIdx in range(len(self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx])):
                partition_func = 0
                for aspect in self.TopicModel:
                    for word in self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx]:
                        partition_func += self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx][word] * (1 - self.HPB[rvwIdx][lineIdx][word]) * \
                                 self.HP[rvwIdx][lineIdx][word][aspect]

                for aspect in self.TopicModel:
                    nom = 0
                    for word in self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx]:
                        nom += self.word_frequency_ByRvwIdxSentenceIdx[rvwIdx][lineIdx][word] * (1 - self.HPB[rvwIdx][lineIdx][word]) * \
                               self.HP[rvwIdx][lineIdx][word][aspect]
                    try:
                        self.PI[rvwIdx][lineIdx][aspect] = nom / partition_func
                    except:
                        print(rvwIdx, lineIdx, aspect, nom, partition_func)


    def delta_param(self, previous_PI, PI):

        delta = 0.0
        for reviewNum in range(len(self.word_frequency_ByRvwIdxSentenceIdx)):
            for lineNum in range(len(self.word_frequency_ByRvwIdxSentenceIdx[reviewNum])):
                for aspect in self.TopicModel:
                    delta += math.pow(PI[reviewNum][lineNum][aspect] - previous_PI[reviewNum][lineNum][aspect], 2)
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
            for i in range(len(self.word_frequency_ByRvwIdxSentenceIdx))]
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

    def save_score(self):
        Precision = []
        Recall = []
        F_Measure = []
        GlobalNDCG = []

        Precision.append([])
        Recall.append([])
        F_Measure.append([])
        GlobalNDCG.append([])

        # for MyK in range(0,len(ExplicitFeatures)+1):
        MyK = 0
        for ColorProbability in np.arange(0.35, 0.39, 0.05):
            Theta = 0.0
            TP = 0
            FP = 0
            FN = 0
            FrequentThreshold = 0.0
            ndcgList = []

            for reviewNum in range(0, len(self.word_frequency_ByRvwIdxSentenceIdx)):
                for lineNum in range(0, len(self.word_frequency_ByRvwIdxSentenceIdx[reviewNum])):
                    if '[u]' in self.raw_reviews[reviewNum][lineNum]:
                        ActualImplicitFeatureSet = set([])
                        ActualExplicitFeatureSet = set([])
                        for item in self.groundtruth[reviewNum][lineNum]:
                            if item.implicit == True:
                                ActualImplicitFeatureSet.add(item.aspect)
                            else:
                                ActualExplicitFeatureSet.add(item.aspect)

                        InferredImplicitFeatureSet = set([])

                        for aspect in sorted(self.PI[reviewNum][lineNum], key=self.PI[reviewNum][lineNum].get, reverse=True):
                            if self.PI[reviewNum][lineNum][aspect] >= ColorProbability \
                                    and aspect not in ActualExplicitFeatureSet \
                                    and aspect not in self.word_frequency_ByRvwIdxSentenceIdx[reviewNum][lineNum] \
                                    and self.ExplicitFeatures[aspect] >= FrequentThreshold \
                                    and aspect not in InferredImplicitFeatureSet:
                                # print(word, aspect, HP[reviewNum][lineNum][word][aspect],HPB[reviewNum][lineNum][word])
                                InferredImplicitFeatureSet.add(aspect)


                        for aspect in self.ExplicitFeatures:
                            if self.ExplicitFeatures[aspect] < FrequentThreshold or aspect in self.word_frequency_ByRvwIdxSentenceIdx[reviewNum][lineNum]:
                                try:
                                    ActualImplicitFeatureSet.remove(aspect)
                                except:
                                    emni = 1

                        if len(ActualImplicitFeatureSet) == 0:
                            continue

                        TP += len(InferredImplicitFeatureSet.intersection(ActualImplicitFeatureSet))
                        FP += len(InferredImplicitFeatureSet - ActualImplicitFeatureSet)
                        FN += len(ActualImplicitFeatureSet - InferredImplicitFeatureSet)

                        if len(ActualImplicitFeatureSet - InferredImplicitFeatureSet) > 0:
                            print("Missd:",ActualImplicitFeatureSet,"\tSen:",self.raw_reviews[reviewNum][lineNum])


            try:
                Precision[-1].append(TP / (TP + FP))
            except:
                Precision[-1].append(0)
            try:
                Recall[-1].append(TP / (TP + FN))
            except:
                Recall[-1].append(0)
            try:
                F_Measure[-1].append(
                    (2 * Precision[-1][MyK] * Recall[-1][MyK]) / (
                                Precision[-1][MyK] + Recall[-1][MyK]))
            except:
                F_Measure[-1].append(0)

            print('P==' + str(ColorProbability))
            print(Precision[-1][MyK], Recall[-1][MyK], F_Measure[-1][MyK])
            MyK += 1


        with open(Results_Directory + 'Summary.csv', 'w') as Summaryfile:
            Summaryfile.write('P,Precision,Recall,F_measure,NDCG\n')
            col_avg_Precision = [sum(x) / len(Precision) for x in zip(*Precision)]
            col_avg_Recall = [sum(x) / len(Recall) for x in zip(*Recall)]
            col_avg_F_Measure = [sum(x) / len(F_Measure) for x in zip(*F_Measure)]

            for i in range(0, len(col_avg_Precision)):
                Summaryfile.write(str(i * 0.05) + ',' + str(col_avg_Precision[i]) + ',' + str(col_avg_Recall[i]) + ',' + str(
                    col_avg_F_Measure[i]) + '\n')

equ_file_path = Data_Directory + EquivalenceFile
stopword_dir = Data_Directory+'lemur.stopwords'
data_dir = Data_Directory + DataFile

rp = ReviewPreprocessor()
stopword = rp.stopword_from_file(stopword_dir)
punctuation = "'.,:;!?1234567890"
equ_aspect = rp.equivalent_aspects_from_file(equ_file_path)
n,r = rp.review_lines_from_file(data_dir)
trainData = rp.train_data_From_reviews(r)
stat = FeatureMiningSentenceLevel(stopword, punctuation, equ_aspect, trainData)
max_iter = 50
lambdaB = 0.7
dist_threshold = 1e-6
stat.main(max_iter,lambdaB,dist_threshold)
stat.save_score()

#print(r)
#print(td)