# coding: utf-8
import csv
import numpy as np
import pandas as pd
import pdb
import os

root = '../../'
writer = csv.writer(open('../../DKT_atoms.csv','w',newline=''))
imputed_list = []
for dirpath,dirnames,filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith('_imputed.csv'):
            imputed_list.append(filename)

problem_ID = 0 # make each question to be an ID
problem_set = {} # unique problem type
answer_set = {'CORRECT':1, 'INCORRECT':0}
print ('The num of students is ', len(filename))

for imputed_file in imputed_list:

    student = pd.read_csv(root + imputed_file, low_memory=False)
    Response_null = pd.isnull(student["Response"])
    student_res = student[['question','Response']][Response_null == False]
    student_res = student_res[student_res['question'] != '_root'][student_res['question'] != 'done'][student_res['question'] != 'nl'][student_res['Response'] != 'HINT']
    problem_IDs = [] # the total problems sequence of each student.
    answers = [] # the total answers sequence of each student

    for ix, row in student_res.iterrows():
        # print (row['Response'],row['question'])
        if not row['question'] in problem_set:
            problem_ID += 1
            problem_set.update({row['question']: problem_ID})
        problem_IDs.append(problem_set[row['question']])
        answers.append(answer_set[row['Response']])
    num_problem = len(problem_IDs)
    print ('num_students',len(filename),'\n num_problem',num_problem,'\n len(problem_IDs) ',len(problem_IDs),'\n len(answers)', len(answers))

    writer.writerow([num_problem])
    writer.writerow(problem_IDs)
    writer.writerow(answers)






