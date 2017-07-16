# coding: utf-8
import csv
import numpy as np
import pandas as pd
import pdb
import os
import pdb
root = '../../'
writer = csv.writer(open(root + 'DKT_atoms.csv','w',newline=''))
imputed_list = []
for dirpath,dirnames,filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith('_imputed.csv'):
            imputed_list.append(dirpath + '/' + filename)


problem_ID = 0 # make each question to be an ID
problem_set = {} # unique problem type
answer_set = {'CORRECT':1, 'INCORRECT':0}
num_student = len(imputed_list)
print ('The num of students is ', num_student)
for imputed_file in imputed_list:

    student = pd.read_csv(imputed_file, low_memory=False)
    Response_null = pd.isnull(student["Response"])
    student_res = student[['Screen_Number','Representation_Number_Within_Screen','question','Response']][Response_null == False]
    student_res = student_res[student_res['question'] != '_root'][student_res['question'] != 'done'][student_res['question'] != 'nl'][student_res['Response'] != 'HINT']
    problem_IDs = [] # the total problems sequence of each student.
    answers = [] # the total answers sequence of each student
    filter_student = False
    if len(student['Screen_Number'].unique()) == 1:
        filter_student = True
        
    for ix, row in student_res.iterrows():
        if row['Screen_Number']>3 :
            break
        if str(row['Representation_Number_Within_Screen'])== 'nan':
            filter_student = True
            break
        SPquestion = str(row['Screen_Number']) + str(row['Representation_Number_Within_Screen']) + row['question']
        # print (SPquestion)
        if not row['question'] in problem_set:
            problem_ID += 1
            problem_set.update({row['question']: problem_ID})
        problem_IDs.append(problem_set[row['question']])
        answers.append(answer_set[row['Response']])
    num_problem = len(problem_IDs)
    if filter_student == True: # Screen or page number missing in this student so we filter it out
        print (imputed_file,'filtering out, screen/page info missing')
        num_student -= 1
        continue
    writer.writerow([num_problem])
    writer.writerow(problem_IDs)
    writer.writerow(answers)
    print ('num_problem',num_problem,'\n len(problem_IDs) ',len(problem_IDs),'\n len(answers)', len(answers))
print ('num_students',num_student)






