import os
import csv

from upsit import plt,cumul_hist

def load():
    """Load Parkinson's Disease Biomarkers Project data."""  

    module_path = os.path.dirname(os.path.realpath(__file__))
    people_path = os.path.join(module_path,'Patient_Status.csv')
    results_path = os.path.join(module_path,'Univ._of_Pennsylvania_Smell_ID_Test.csv')

    people = open(people_path,'r')
    people_reader = csv.reader(people)

    results = open(results_path,'r')
    results_reader = csv.reader(results)

    data = {}
    fields = {'HC':0,
              'PD':1,
              'no image':None,
              'SWEDD':None}

    for row in people_reader:
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

def plot_cumul_hist(data,booklet=None):
    """
    Plots cumulative histogram for PDBP data.
    data: output of load()
    booklet: optionally restrict to one of four booklets (1-4).  
    """
    
    if booklet:
        smell = [value['upsit'][booklet-1] for key,value in data.items()]
    else:
        smell = [sum(value['upsit']) for key,value in data.items()]
    recruitment = [value['recruitment'] for key,value in data.items()]
    
    smell_ctl = [smell[i] for i in range(len(recruitment)) if recruitment[i]==0]
    smell_ctl = sorted(smell_ctl)
    
    smell_pd = [smell[i] for i in range(len(recruitment)) if recruitment[i]==1]
    smell_pd = sorted(smell_pd)

    cumul_hist(smell_ctl,color='k')
    cumul_hist(smell_pd,color='r')

    plt.xlabel('UPSIT score')
    plt.ylabel('Cumulative Probability')

