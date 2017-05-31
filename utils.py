# coding: utf-8

def inputStudent(csvInput):
    for line in csvInput:
        # nStep, questionsID, correct = yield(line)
        if line[-1]=='':
            yield line[:-1]
        else: yield line