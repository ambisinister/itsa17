"""
This file generates breakdowns from a folder of smfiles
"""

import os
import pickle

def process_sm(smpath):
    smfile = open(smpath, encoding='utf-8-sig').read()

    # Dealing with bpms is a surprising pain with changes, so I'll do the lazy method
    bpm = smfile.split('#TITLE:')[1][6:9]
    #bpm = smfile.split('#BPMS:')[1].split(';')[0]
    #bpm = bpm.split(',')[0].split('=')[1]

    notes = smfile.split('#NOTES:')
    header = notes[0]
    chart = notes[1]
    
    chartnotes = chart.split('\n')
    listed_breakdown = chartnotes[2][:-1]
    rating = chartnotes[4][:-1].strip()
    breakdown = []

    steps = chartnotes[6:]

    streaming = False
    currentrun = 0
    steps_in_measure = 0
    for step in steps:
        if ',' in step:
            if streaming is True and steps_in_measure >= 16:
                currentrun += 1
                steps_in_measure = 0
            elif streaming is True and steps_in_measure < 16:
                streaming = False
                breakdown.append(currentrun)
                currentrun = -1
                steps_in_measure = 0
            elif streaming is False and steps_in_measure >= 16:
                streaming = True
                breakdown.append(currentrun)
                currentrun = 1
                steps_in_measure = 0
            elif streaming is False and steps_in_measure < 16:
                currentrun -= 1
                steps_in_measure = 0
        else:
            if step.count('1') > 0:
                steps_in_measure += 1

    # End with stream case
    if streaming is True:
        breakdown.append(currentrun)

    # Crop leading and trailing break, which do not contribute to difficulty
    while breakdown[0] < 0:
        breakdown = breakdown[1:]
    while breakdown[-1] < 0:
        breakdown = breakdown[:-1]

    processed = {
        "breakdown": breakdown,
        "listed breakdown": listed_breakdown,
        "bpm": bpm,
        "rating": rating
    }
    
    return processed
    


def process_folder(songspath):
    data = []
    for fold in os.listdir(songspath):
        if os.path.isdir(songspath+fold): 
            print(fold)
            for fil in os.listdir(songspath+fold):
                if fil.endswith('.sm'):
                    data.append(process_sm(songspath+fold+'/'+fil))

    return data




if __name__ == '__main__':
    songspath = '/home/ambi/.stepmania-5.0/Songs/'
    train = [
        'East Coast Stamina 6 Qualifiers/',
        'ECS7 Qualifiers/',
        'Stamina RPG 3/',
        'Stamina RPG 4/'
    ]
    val = 'Stamina RPG 5/'
    test = 'Stamina RPG 6/'

    # Train
    dataset = []
    for t in train:
        t_set = process_folder(songspath+t)
        dataset = [*dataset, *t_set]
    pickle.dump(dataset, open('./train.pkl', 'wb'))
    
    # Validation
    dataset = process_folder(songspath+val)
    pickle.dump(dataset, open('./val.pkl', 'wb'))

    # Test
    dataset = process_folder(songspath+test)
    pickle.dump(dataset, open('./test.pkl', 'wb'))
