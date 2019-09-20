from pprint import pprint
import random
import fasttext
import csv
import re
import os

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

#### Data pipeline ####

csv_trump = open('Trump.csv')
csv_hillary = open('Hillary.csv')
csv_obama = open('Obama.csv')

#### Extract stop words ###

f = open('stopWords.txt', 'r')
line = f.readline()
stopWords = []
while line:
    stopWords.append(line.replace('\n', ''))
    line = f.readline()

# Delete all Trump tweet from TrumpHilary file

# Hillary

reader = csv.reader(csv_hillary, delimiter=',')
cnt = 0
hillaryData = []
for row in reader:
    if cnt != 0:
        if row[1] == 'HillaryClinton' and row[3] == 'False':
            formatedRow = row[2].replace('\n', ' ')
            formatedRow = re.sub(r'https?:\/\/.*[\r\n]*', '', formatedRow, flags=re.MULTILINE)
            hillaryData.append("__label__Hillary " + formatedRow)
    cnt += 1

print('Hillary: ' + str(len(hillaryData)))
# print(hillaryData)

# Trump

reader = csv.reader(csv_trump, delimiter=',')
cnt = 0
trumpData = []
for row in reader:
    if cnt != 0:
        formatedRow = row[2].replace('\n', ' ')
        formatedRow = re.sub(r'https?:\/\/.*[\r\n]*', '', formatedRow, flags=re.MULTILINE)
        trumpData.append("__label__Trump " + formatedRow)
    cnt += 1

print('Donald: ' + str(len(trumpData)))

# Obama

reader = csv.reader(csv_obama, delimiter=',')
cnt = 0
obamaData = []
for row in reader:
    if cnt != 0:
        formatedRow = row[2].replace('\n', ' ')
        formatedRow = re.sub(r'https?:\/\/.*[\r\n]*', '', formatedRow, flags=re.MULTILINE)
        obamaData.append("__label__Obama " + formatedRow)
    cnt += 1

print('Obama: ' + str(len(obamaData)))


# Format each file : __label__{name} {tweet} into a single file twweets.txt
array = obamaData + trumpData + hillaryData
random.shuffle(array)

separator = 12000

# Split into two files One to train, one to validate
data = open('supervised_data.txt', 'w')
textData = '\n'.join(array[:separator])

dataValidation = open('supervised_valid.txt', 'w')
textDataValidation = '\n'.join(array[separator:])

for word in stopWords:
    textData.replace(word, '')
    textDataValidation.replace(word, '')

data.write(textData)
dataValidation.write(textDataValidation)

os.system(r"""cat supervised_data.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > supervised_data_preprocessed.txt""")
os.system(r"""cat supervised_valid.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > supervised_valid_preprocessed.txt""")

# Train the algorithm

classifier = fasttext.train_supervised("supervide_data_preprocessed.txt", lr=0.3, wordNgrams=2)

print_results(*classifier.test('supervised_valid_preprocessed.txt'))

#print(f'Our labels {classifier.labels}')

# Make prediction

print(classifier.predict('The New York Times is now blaming an editor for the horrible mistake they made in trying to destroy or influence Justice Brett Kavanaugh. It wasn’t the editor, the Times knew everything. They are sick and desperate, losing in so many ways!', k=3))

print(classifier.predict('A terrible truth: Not everyone who is eligible to vote in America has free and fair access to the ballot box. Hear @EricHolder, @AriBerman, @AndrewGillum, & @MariaTeresa1  lay out how we can defend voting rights: # DefenseOfDemocracy https: // youtube.com/watch?v=2X0DFP67nGE', k=3))

print(classifier.predict('The New York Times is at its lowest point in its long and storied history. Not only is it losing a lot of money, but it is a journalistic disaster, being laughed at even in the most liberal of enclaves. It has become a very sad joke all all over the World. Witch Hunt hurt them...', k=3))

print(classifier.predict('We are all united by the same love of Country, the same devotion to family, and the same profound faith that America is blessed by the eternal grace of ALMIGHTY GOD! Bound by these convictions, we will campaign for every vote & we will WIN the Great State of NEW MEXICO in 2020!', k=3))

print(classifier.predict('“Putting people who have not been convicted of a crime in  # SolitaryConfinement is just wrong.” #Solitary #Immigrants #immoral #humanrights #socialjustice #ICE https: // icij.org/investigations/solitary-voices/solitary-confinement-its-immoral-its-unethical-its-torture / via @ICIJorg', k=3))
