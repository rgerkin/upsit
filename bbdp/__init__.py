import os
import sys
from datetime import datetime,timedelta

import numpy as np
import xlrd

filepath = os.path.abspath(__file__)
dirpath = filepath
for i in range(3):  
  dirpath = os.path.dirname(dirpath)
  sys.path.append(dirpath)

from upsit import Subject,Question,Response,QuestionSet,ResponseSet,Test,plt

def load(kind):
	"""Load Banner Brain and Body Donation Project data."""  

	module_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(module_path,'data')
	questions = []
	test_path = os.path.join(data_path,'GerkinSmithUPSITautopsy9_10_14.xlsx')
	test_wb = xlrd.open_workbook(test_path)
	test_key = test_wb.sheet_by_name('smellTestKey')
	  
	for q in range(1,41):
		row = test_key.row_values(q)
		options = row[1:5] # 4 possible options
		answer_num = int(row[6]-1) # Change from 0-indexed to 1-indexed.  
		questions.append(Question(options,answer_num))
	question_set = QuestionSet(questions)

	if kind == 'dugger':
		disease_path = os.path.join(data_path,
									'GerkinSmithUPSITautopsy9_10_14.xlsx')
		ctrl_path = os.path.join(data_path,
								 'GerkinSmithQueryControls9_17_14.xlsx')
		cp_path = os.path.join(data_path,'Clinicopathological Correlations.xls')

		disease_wb = xlrd.open_workbook(disease_path)
		ctrl_wb = xlrd.open_workbook(ctrl_path)
		cp_wb = xlrd.open_workbook(cp_path)
	
		pd_sheet = disease_wb.sheet_by_name('"pure"PDonly1test')
		pd_subjects,pd_tests = parse_tests_dugger(kind,pd_sheet,question_set,
										 subject_label='pd')

		ctrl_sheet = ctrl_wb.sheet_by_name('AllNPcontrolVisits')
		ctrl_subjects,ctrl_tests = parse_tests_dugger(kind,ctrl_sheet,
											 question_set,subject_label='ctrl')  

		# Currently not used.  
		cp = cp_wb.sheets()[0] # Only one sheet.      
	  
		subjects = ctrl_subjects.copy()
		subjects.update(pd_subjects)
		tests = pd_tests + ctrl_tests
	  
	elif kind == 'hentz':
		all_path = os.path.join(data_path,'D20Mar2015a.xls')
		all_wb = xlrd.open_workbook(all_path)
		all_sheet = all_wb.sheet_by_name('Data')

		subjects,tests = parse_tests_hentz(kind,all_sheet,question_set) 
	
	return subjects,tests
	
def parse_tests_hentz(kind,tests_sheet,question_set,subject_label=None):
	"""Parse a worksheet of tests to return subject and tests.""" 

	subjects = {}
	tests = []
	
	headers = tests_sheet.row_values(0)
	hd = {key:i for i,key in enumerate(headers)}
	for row_num in range(1,tests_sheet.nrows):
		row = tests_sheet.row_values(row_num)

		case_id = row[hd['shri_case_num']]
		if case_id not in subjects:
			subject = Subject(case_id)
			# Age is reported as an integer or as 100+.   
			expired_age = row[hd['deathage']]
			subject.expired_age = 100 if expired_age=='100+' else int(expired_age)
			# Gender is reported as 1 or 2.  
			subject.gender = row[hd['female']]
			if subject_label is not None:
				subject.label = subject_label
			subject.dementia = 2 in row[16:25]
			subject.stint = float(row[hd['stint']])
			subject.other = [int(_)>0 for _ in row[5:17]+row[19:25]]
			subject.label = 'ctrl' if int(row[hd['controlp']]) else 'other'
			subjects[case_id] = subject
		  
		responses = {}
		for q in range(1,41):
			choice_num = row[hd['smell_%d' % q]] 
			if type(choice_num) is float:
				choice_num = int(choice_num)-1 # Change to 0-indexed.  
			else:
				choice_num = None
			responses[q] = Response(question_set.questions[q],choice_num)
	  
		response_set = ResponseSet(responses,indices=responses.keys())
		test = Test(subjects[case_id],response_set,None)
		tests.append(test)    

	return subjects,tests


def parse_tests_dugger(kind,tests_sheet,question_set,subject_label=None):
	"""Parse a worksheet of tests to return subject and tests.""" 

	subjects = {}
	tests = []
	
	headers = tests_sheet.row_values(0)
	hd = {key:i for i,key in enumerate(headers)}
	for row_num in range(1,tests_sheet.nrows):
		row = tests_sheet.row_values(row_num)

		case_id = row[hd['CaseID']]
		if case_id not in subjects:
			subject = Subject(case_id)
			# Age is reported as an integer or as 100+.   
			expired_age = row[hd['expired_age']]
			subject.expired_age = 100 if expired_age=='100+' else int(expired_age)
			# Gender is reported as 1 or 2.  
			subject.gender = row[hd['tbl_donors.gender']]-1
			if subject_label is not None:
				subject.label = subject_label
			subject.demented = row[hd['dementia_nos']] in [1,'yes']
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
			responses[q] = Response(question_set.questions[q],choice_num)
		
		response_set = ResponseSet(responses,indices=responses.keys())
		test = Test(subjects[case_id],response_set,test_date)
		tests.append(test)    

	return subjects,tests

def correct_matrix(tests, kind=None):
	correct = {}
	for test in tests:
		if (kind is None) or (test.subject.label == kind):
			correct[test.subject.case_id]= [int(test.response_set.responses[i].correct) \
									for i in range(1,41)]
			print(test.subject.case_id,test.response_set.responses[35].correct)
	
	return np.array(correct.values())

def correct_corrs(tests, kind=None):
	matrix = correct_matrix(tests, kind=kind)
	for test in tests:
		if (test.subject.label is None) or (test.subject.label == kind):
			correct[test.subject]= [int(test.response_set.responses[i].correct) \
									for i in range(1,41)]

	corrs = np.corrcoef(matrix.transpose())
	
	plt.figure()
	plt.pcolor(np.arange(0.5,41.5,1),np.arange(0.5,41.5,1),corrs,cmap='RdBu_r',vmin=-1,vmax=1)
	plt.colorbar()
	plt.xlim(0.5,40.5)
	plt.ylim(0.5,40.5)

	return corrs

def factor_analysis(tests):
	from sklearn.decomposition import FactorAnalysis
	from sklearn.cross_validation import cross_val_score
	
	matrix = correct_matrix(tests,kind='ctrl')
	print(matrix.shape)
	# matrix must have a number of rows divisible by 3.  
	# if it does not, eliminate some rows, or pass cv=a to cross_val_score,
	# where 'a' is a number by which the number of rows is divisible.  
	fa = FactorAnalysis()
	fa_scores = []
	n_components = np.arange(1,41)
	for n in n_components:
		fa.n_components = n
		fa_scores.append(np.mean(cross_val_score(fa, matrix)))

	plt.plot(n_components,fa_scores)
	
	return n_components,fa_scores
	
