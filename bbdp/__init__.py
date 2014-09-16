import os
from datetime import datetime,timedelta

import xlrd

from upsit import Subject,Question,Response,QuestionSet,Test

def load():
    """Load Banner Brain and Body Donation Project data."""  

    module_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(module_path,'data')
    upsit_path = os.path.join(data_path,'GerkinSmithUPSITautopsy9_10_14.xlsx')
    cp_path = os.path.join(data_path,'Clinicopathological Correlations.xls')

    upsit_wb = xlrd.open_workbook(upsit_path)
    cp_wb = xlrd.open_workbook(cp_path)
    
    upsit_tests = upsit_wb.sheet_by_name('"pure"PDonly1test')
    upsit_smelltestkey = upsit_wb.sheet_by_name('smellTestKey')
    
    cp = cp_wb.sheets()[0] # Only one sheet.  

    questions = {}
    for q in range(1,41):
        row = upsit_smelltestkey.row_values(q)
        options = row[1:5] # 4 possible options
        answer_num = int(row[6]-1) # Change from 0-indexed to 1-indexed.  
        questions[q] = Question(q,options,answer_num)

    subjects = {}
    tests = []
    gender = {1:'M',2:'F'}
    headers = upsit_tests.row_values(0)
    hd = {key:i for i,key in enumerate(headers)}
    for row_num in range(1,upsit_tests.nrows):
        row = upsit_tests.row_values(row_num)
        
        case_id = row[hd['CaseID']]
        if case_id not in subjects:
            subject = Subject(case_id)
            # Age is reported as an integer.  
            subject.expired_age = int(row[hd['expired_age']])
            # Gender is reported as 1 or 2.  
            subject.gender = row[hd['tbl_donors.gender']]
            subjects[case_id] = subject
        
        test_date = row[hd['smell_test_date']]
        test_date = datetime(1900,1,1) + timedelta(int(test_date)-2)

        responses = {}
        for q in range(1,41):
            choice_num = row[hd['smell_%d' % q]] 
            if type(choice_num) is float:
                choice_num = int(choice_num)-1 # Change to 0-indexed.  
            else:
                choice_num = None
            responses[q] = Response(questions[q],choice_num)
        
        test = Test(subjects[case_id],responses.values(),test_date)
        tests.append(test)        
    
    #data = {key:value for key,value in data.items() if 'upsit' in value}
    return subjects,tests