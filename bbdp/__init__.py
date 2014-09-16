import os
import xlrd

from upsit import plt,cumul_hist

def load():
    """Load Banner Brain and Body Donation Project data."""  

    module_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(module_path,'data')
    upsit_path = os.path.join(data_path,'GerkinSmithUPSITautopsy9_10_14.xlsx')
    cp_path = os.path.join(data_path,'Clinicopathological Correlations.xls')

    upsit_wb = xlrd.open_workbook(upsit_path)
    cp_wb = xlrd.open_workbook(cp_path)
    
    upsit_tests = upsit_wb.sheet_by_name('allPDtests')
    upsit_smelltestkey = upsit_wb.sheet_by_name('smellTestKey')
    
    cp = cp_wb.sheets()[0] # Only one sheet.  

    data = {}
    fields = {}
    gender = {1:'M',2:'F'}

    upsit_key = []
    for q in range(1,41):
        row = upsit_smelltestkey.row_values(q)
        options = row[1:5] # 4 possible options
        answer_num = int(row[6]-1) # Change from 0-indexed to 1-indexed.  
        answer = options[answer_num-1]
        upsit_key += [(options,answer_num)]

    headers = upsit_tests.row_values(0)
    hd = {key:i for i,key in enumerate(headers)}
    for row_num in range(1,upsit_tests.nrows):
        row = upsit_tests.row_values(row_num)
        case_id = row[hd['CaseID']]
        if case_id not in data:
            data[case_id] = {
                # Age is reported as an integer.  
                'expired_age':int(row[hd['expired_age']]), 
                # Gender is reported as 1 or 2.  
                'gender':gender[row[hd['tbl_donors.gender']]], 
                'upsit':[]
                }
        # 40 questions. 
        responses = [row[hd['smell_%d' % q]] for q in range(1,41)] 
        # Change to 0-indexed.  
        responses = [int(x)-1 if type(x) is float else None for x in responses]
        data[case_id]['upsit'].append(responses)

    #data = {key:value for key,value in data.items() if 'upsit' in value}
    return upsit_key,data