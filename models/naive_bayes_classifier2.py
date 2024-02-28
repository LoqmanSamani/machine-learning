import pandas as pd


""" A naive Bayes classifier for categorical data (email spam detector)."""



class SpamDetector(object):

    def __init__(self):

        self.word_freq = {}
        self.class_freq = {}



    def processing(self, text):

        text = text.lower()
        words = text.split()
        words = set(words)

        return list(words)




    def word_class_freq(self, data):
        # data is a dataframe
        # data.columns = (text, spam)

        freqs = {}  # a dict to store the frequency of each word
        # freqs = {word1: {"ham": 2, "spam": 4}, ...}

        for _, row in data.iterrows():

            for word in row['text']:
                if word not in freqs:
                    freqs[word] = {'spam': 0, 'ham': 0}

                if row['spam'] == 1:

                    freqs[word]['spam'] += 1
                else:
                    freqs[word]['ham'] += 1

        return freqs




    def class_freqs(self, data):

        freqs = {}
        l = data["spam"].values.tolist()

        freqs["spam"] = l.count(1)
        freqs["ham"] = l.count(0)

        return freqs





    def train(self, data):

        data["text"] = data["text"].apply(self.processing)
        self.word_freq = self.word_class_freq(data)
        self.class_freq = self.class_freqs(data)





    def predict(self, text):

        text = self.processing(text)

        cph = 1  # cumulative product ham
        cps = 1  # cumulative product spam

        for word in text:
            if word in self.word_freq.keys():

                # calculate p(w|s)
                prob_w_s = self.word_freq[word]["spam"] / self.class_freq["spam"]
                cps *= prob_w_s

                # calculate p(w|h)
                prob_w_h = self.word_freq[word]["ham"] / self.class_freq["ham"]
                cph *= prob_w_h

        if (cps * self.class_freq["spam"]) + (cph * self.class_freq["ham"]) != 0:
            spam_prob = (cps * self.class_freq["spam"]) / ((cps * self.class_freq["spam"]) + (cph * self.class_freq["ham"]))
        else:
            spam_prob = 0.5

        is_spam = (1 if spam_prob > 0.5 else 0)

        return is_spam, spam_prob







path = "/home/sam/Documents/projects/machine_learning/data/emails.csv"

with open(path, "r") as file:
    data = pd.read_csv(file)

print(data.head())
"""
                                               text  spam
0  Subject: naturally irresistible your corporate...     1
1  Subject: the stock trading gunslinger  fanny i...     1
2  Subject: unbelievable new homes made easy  im ...     1
3  Subject: 4 color printing special  request add...     1
4  Subject: do not have money , get software cds ...     1
"""


print(len(data))
""" 5728 """



# split the dataset into train and test
train_data = data[:5000]
test_data = data[5000:]



classifier = SpamDetector()
classifier.train(train_data)



print(classifier.class_freq)
""" {'spam': 1368, 'ham': 3632} """



y_test = test_data["spam"].values.tolist()
x_test = test_data["text"].values.tolist()



print(len(y_test))
print(len(x_test))
"""
728
728
"""



y_probs = []
y_hats = []

for text in x_test:

    is_spam, spam_prob = classifier.predict(text)
    y_probs.append(spam_prob)
    y_hats.append(is_spam)



accuracy = (sum([1 if y == y_hat else 0 for y, y_hat in zip(y_test, y_hats)]) / len(y_test)) * 100


print(f"The model accuracy based on test data is: {accuracy} % .")

"""
The model accuracy based on test data is: 99.86263736263736 % .
"""








