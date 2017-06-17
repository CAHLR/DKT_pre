# coding: utf-8
import csv
import numpy as np
import utils
import pdb
max_train = None
max_steps = None

class DataAssistMatrix():
    def __init__(self):
        print('Loading data...')
        #training process
        root = '../data/assistments/'
        trainPath = root + 'builder_train.csv'
        csvFile = open(trainPath, 'r')
        csvInput = csv.reader(csvFile)
        count = 0
        trainData = []
        #self.train_longest = 0
        #self.test_longest = 0
        self.longest = 100
        '''
        we assume self.questions is useless, we only need self.n_questions
        '''
        self.questions = []
        self.n_questions = 0
        totalAnswers = 0
        self.max_questionID = 0

        while(True):
            student = self.loadStudent(csvInput)
            if student == None:
                print ('Load student failed !')
                break
            if(student.n_answers >= 2 and student.n_answers<=self.longest):
                trainData.append(student)
            #if len(trainData) % 100 == 0:
                #print 'The length of train data is now ',trainData
            #if student.n_answers > self.train_longest:
             #   self.train_longest = student.n_answers

            totalAnswers = totalAnswers + student.n_answers
        self.trainData = trainData
        csvFile.close()

        #testing processing
        testPath = root + 'builder_test.csv'
        csvFile = open(testPath,'r')
        csvInput = csv.reader(csvFile)

        count = 0
        testData = []
        self.questions = []
        while(True):
            student = self.loadStudent(csvInput)
            if student == None:
                print ('Load student failed or finished')
                break
            if(student.n_answers >= 2 and student.n_answers<=self.longest):
                testData.append(student)
            #if len(testData) % 100 == 0:
                #print 'The length of test data is now ',testData
            #if student.n_answers > self.test_longest:
                #self.test_longest = student.n_answers
            totalAnswers = totalAnswers + student.n_answers
        self.testData = testData
        csvFile.close()
        print('total answers', totalAnswers)
        #print('longest train data is ', self.train_longest)
        #print('longest test data is ',self.test_longest)
        print ('longest data is',self.longest)
        print('max questionsID', self.max_questionID)


    def loadStudent(self, csvInput):
        try:
            nStep = next(utils.inputStudent(csvInput))
            questionsID = next(utils.inputStudent(csvInput))
            correct = next(utils.inputStudent(csvInput))
        except:
            print ('execption loadStudent')
            return None
        # if nStep == None or questionsID == None or correct == None:
        #     return None
        n = int(nStep[0])
        if(max_steps != None):
            n = max_steps
        for i in  range(len(questionsID)):
            if questionsID[i] not in self.questions:
                self.questions.append(questionsID[i])
                self.n_questions = self.n_questions + 1
        stu = student(n,questionsID,correct)
        if max(stu.questionsID) > self.max_questionID:
            self.max_questionID = max(stu.questionsID)
        return stu

class student():
    def __init__(self,n,questionsID,correct):
        self.n_answers = n
        self.questionsID = np.zeros(n,int)
        self.correct = np.zeros(n,int)
        for i in  range(n):
            if i > n:
                break
            self.questionsID[i] = int(float(questionsID[i])) # we don't plus 1 as the origin code
        for i in  range(n):
            if i > n:
                break
            self.correct[i] = int(float(correct[i]))




