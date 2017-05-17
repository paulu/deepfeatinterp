import os, sys
import csv
import pickle
import numpy as np
from sklearn.svm import LinearSVC

fields = [
  '5_o_Clock_Shadow',
  'Arched_Eyebrows',
  'Attractive',
  'Bags_Under_Eyes',
  'Bald',
  'Bangs',
  'Big_Lips',
  'Big_Nose',
  'Black_Hair',
  'Blond_Hair',
  'Blurry',
  'Brown_Hair',
  'Bushy_Eyebrows',
  'Chubby',
  'Double_Chin',
  'Eyeglasses',
  'Goatee',
  'Gray_Hair',
  'Heavy_Makeup',
  'High_Cheekbones',
  'Male',
  'Mouth_Slightly_Open',
  'Mustache',
  'Narrow_Eyes',
  'No_Beard',
  'Oval_Face',
  'Pale_Skin',
  'Pointy_Nose',
  'Receding_Hairline',
  'Rosy_Cheeks',
  'Sideburns',
  'Smiling',
  'Straight_Hair',
  'Wavy_Hair',
  'Wearing_Earrings',
  'Wearing_Hat',
  'Wearing_Lipstick',
  'Wearing_Necklace',
  'Wearing_Necktie',
  'Young',
  'asian',
  'baby',
  'black',
  'brown_eyes',
  'child',
  'color_photo',
  #'eyeglasses2',
  'eyes_open',
  'flash',
  'flushed_face',
  'frowning',
  'fully_visible_forehead',
  'harsh_lighting',
  #'high_cheekbones2',
  'indian',
  'middle_aged',
  'mouth_closed',
  #'mouth_slightly_open2',
  'mouth_wide_open',
  'no_eyewear',
  'obstructed_forehead',
  'outdoor',
  'partially_visible_forehead',
  'posed_photo',
  'round_face',
  'round_jaw',
  'senior',
  'shiny_skin',
  'soft_lighting',
  'square_face',
  'strong_nose_mouth_lines',
  'sunglasses',
  'teeth_not_visible',
  'white',
]

def load():
    clf = {}
    for field in fields:
        with open('classifiers/%s.pkl' % field, 'rb') as f:
            clf[field] = pickle.load(f)
    return clf

def predict(clf, features):
    result = np.empty(len(fields), dtype=np.float64)
    for i,field in enumerate(fields):
        result[i] = clf[field].decision_function(features)
    return result

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python get_attrs.py <folder> <field>')
	print('Fields: \n')

	for field in fields:
            print(field)

    else:
        folder = sys.argv[1]
        field = sys.argv[2]

        if not field in fields:
            raise Exception('Invalid field')
        field_index = fields.index(field)
        
        with open(os.path.join(folder, 'features.csv')) as f:
            reader = csv.reader(f)
            basenames = []
            featuresTable = []

            for row in reader:
                basenames.append(row[0])
                featuresTable.append(np.array([[float(feature) for feature in row[1:]]]))

            features = np.concatenate(featuresTable, axis = 0)
        print 'features', features.shape

        with open('classifiers/%s.pkl' % field, 'rb') as f:
            clf = pickle.load(f)
            scores = clf.decision_function(features)

        for basename, score in zip(basenames, scores):
            print('%s: %.2f' % (basename, score))
