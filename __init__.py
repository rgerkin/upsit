import csv
from types import NoneType

import numpy as np
import matplotlib.pyplot as plt

#############
# Functions #
#############

def cumul_hist(values,color='r'):
    sorted_values = sorted(values)
    quantiles = np.arange(0,1,1.0/len(values))
    plt.plot(sorted_values,quantiles,color)
    
###########
# Classes #
###########

class Subject(object):
    """
    A subject (person) enrolled in the study.
    """
    def __init__(self,case_id,**kwargs):
        self.case_id = case_id
        for key,value in kwargs:
            setattr(self,key,value)

    donor_id = None
    expired_age = None # Age at death.  

    @property
    def pd_only(self):
        return None

class Question(object):
    """
    One question on an UPSIT test.  
    """

    def __init__(self, num, ordered_choices, correct_index):
        """
        num: question number in a test booklet.  
        ordered_choices: list with 4 strings.  
        correct_index: index in that list of the correct answer. 
        """ 
        
        self.num = num
        self.choices = ordered_choices
        self.correct_index = correct_index
        self.correct = self.choices[correct_index]

class QuestionSet(object):
    """
    A ordered set of questions (i.e. a complete, unanswered test).
    """

    def __init__(self,questions):
        self.questions = questions

class Response(object):
    """
    A specific answer to a question on an UPSIT test.  
    """

    def __init__(self,question,choice):
        """
        question: The question
        response: The index of the response in question.choices, or None.  
        """

        self.question = question
        if type(choice) is NoneType:
            self.choice = choice
            self.choice_num = None
        elif type(choice) is str:
            self.choice = choice
            self.choice_num = question.index(choice)
        elif type(choice) is int:
            self.choice_num = choice
            self.choice = question.choices[choice]
        else:
            raise ValueError("Response must be None, string, or int")

    @property
    def correct(self):
        return self.choice == self.question.correct

    @property
    def blank(self):
        return self.choice is None

class Test(object):
    """
    An ordered set of responses (i.e. a complete, answered test).
    """
    
    def __init__(self,subject,responses,date):
        self.subject = subject
        self.responses = responses
        self.date = date # Date of test as a datetime.datetime

    @property
    def score(self):
        return sum(x.correct for x in self.responses)

    