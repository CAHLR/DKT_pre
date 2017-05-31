# coding: utf-8
import csv
import numpy as np
import utils

max_train = None
max_steps = None

class DataAssistMatrix():
    def __init__(self):
        print('Loading data...')
        #training process
        root = '../data/assistments/'
        trainPath = root + 'builder_train.csv'
        csvfile = file(trainPath, 'rb')
        csvInput = csv.reader(csvfile)
        count = 0
        trainData = []
        longest = 0
        '''
        we assume self.questions is useless, we only need self.n_questions
        '''
        self.questions = []
        self.n_questions = 0
        totalAnswers = 0
        
        # while(True):
        #     student = self.loadStudent(csvInput)
        #     if student == None:
        #         print 'Load student failed !'
        #         break
        #     if(student.n_answers >= 2):
        #         trainData.append(student)
        #     if len(trainData) % 100 == 0:
        #         print 'The length of train data is now ',trainData
        #     if student.n_answers > longest:
        #         #longest = student.n_answers
        #         pass
        # totalAnswers = totalAnswers + student.n_answers
        # self.trainData = trainData
        self.trainData,totalAnswers = self.loadStudent(csvInput,totalAnswers)
        csvfile.close()

        #testing processing
        testPath = root + 'builder_test.csv'
        csvFile = file(testPath,'rb')
        csvInput = csv.reader(csvfile)
        count = 0
        testData = []
        self.questions = []
        # while(True):
        #     student = self.loadStudent(csvInput)
        #     if student == None:
        #         print 'Load student failed !'
        #         break
        #     if(student.n_answers >= 2):
        #         testData.append(student)
        #     if len(testData) % 100 == 0:
        #         print 'The length of test data is now ',testData
        #     if student.n_answers > longest:
        #         pass
        #         #longest = student.n_answers
        #     totalAnswers = totalAnswers + student.n_answers
        # self.testData = testData
        self.testData, totalAnswers = self.loadStudent(csvInput,totalAnswers)
        csvfile.close()
        print('total answers', totalAnswers)
        print('longest', longest)


    def loadStudent(self, csvInput, totalAnswers):
        Data = []

        while(True):
            nStep = utils.inputStudent(csvInput).next()
            questionsID = utils.inputStudent(csvInput).next()
            correct = utils.inputStudent(csvInput).next()
            if nStep == None or questionsID == None or correct == None:
                break
            n = int(nStep)
            if(max_steps != None):
                n = max_steps
            for i in xrange(len(questionsID)):
                if questionsID[i] not in self.questions:
                    self.questions.append(questionsID[i])
                    self.n_questions = self.n_questions + 1           
            Stu = student(n,questionsID,correct)
            if Stu == None:
                print 'Load student failed !'
                break
            if(Stu.n_answers >= 2):
                Data.append(Stu)
            if len(Data) % 100 == 0:
                print 'The length of train data is now ',Data
        totalAnswers = totalAnswers + Stu.n_answers
        return Data, totalAnswers


class student():
    def __init__(self,n,questionsID,correct):
        self.n_answers = n
        self.questionsID = np.zeros(n)
        self.correct = np.zeros(n)
        for i in xrange(len(questionsID)):
            if i > n:
                break
            self.questionsID[i] = questionsID[i] + 1 # why plus 1?
        for i in xrange(len(correct)):
            if i > n:
                break
            self.correct[i] = correct[i]




