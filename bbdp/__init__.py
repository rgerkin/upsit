import os
import xlrd

from upsit import plt,cumul_hist

def load():
    """Load Banner Brain and Body Donation Project data."""  

    module_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(module_path,'data')
    upsit_path = os.path.join(data_path,'GerkinSmithUPSITautopsy9_10_14.xlsx')
    cp_path = os.path.join(data_path,'Clinicopathological Correlations.xlsx')

    upsit_wb = xlrd.open_workbook(uspit_path)
    cp_wb = xlrd.open_workbook(cp_path)
    
    upsit_tests = upsit_wb.sheets[0] # allPDtests
    upsit_smelltestkey = upsit_wb.sheets[2] # smelltestkey
    
    cp = cp_wb.sheets[0] # Only one sheet.  

    data = {}
    fields = {}

    for row_num in range(upsit_tests.nrows):
        row = upsit_purepdonly.row_values()
        case_id = row[0]
        data[case_id] = {
          'expired_age':row[2],
          'gender':row[3],
        }

        try:
            data[int(row[0])] = {'recruitment':fields[row[1]],
                                 'imaging':fields[row[2]]}
        except:
            pass
    for row in results_reader:
        if row[10] == '':
            data[int(row[2])]['upsit'] = [int(row[6]),
                                          int(row[7]),
                                          int(row[8]),
                                          int(row[9])]

    data = {key:value for key,value in data.items() if 'upsit' in value}
    return data