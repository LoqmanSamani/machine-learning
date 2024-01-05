import numpy as np
import matplotlib.pyplot as plt

# Monty Hall Paradox 1

def montyhall1(num_doors=3, choice=2, change=False):

    doors = [i for i in range(1, num_doors + 1)]
    labels = ["Yes"] + ["No" for _ in range(num_doors - 1)]
    np.random.shuffle(labels)

    choices = dict(zip(doors, labels))

    del_num = True
    while del_num:
        num = np.random.choice([1, 2, 3])
        if num != choice and choices[num] != "Yes":
            del_num = False
            del choices[num]

    if change:
        del choices[choice]

    else:
        choices = {choice: choices[choice]}

    return choices


def call_mh1(num_iter=100):

    n1 = []
    n2 = []

    for _ in range(num_iter):

        r1 = montyhall1(num_doors=3, choice=2, change=False)
        r2 = montyhall1(num_doors=3, choice=2, change=True)
        n1.extend(list(r1.values()))
        n2.extend(list(r2.values()))

    prob1 = (len([h for h in n1 if h == "Yes"]) / len(n1)) * 100
    prob2 = (len([h for h in n2 if h == "Yes"]) / len(n2)) * 100

    return prob1, prob2

"""
probs1 = []
probs2 = []

for _ in range(100):

    prob1, prob2 = call_mh1(1000)
    probs1.append(prob1)
    probs2.append(prob2)
    print(prob1)
    print(prob2)


plt.pie([np.mean(probs1), np.mean(probs2)], labels=["Not Change", "Change"], autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Monty Hall Paradox")
plt.show()
"""

"""
32.5
65.5

32.5
67.5

34.3
65.7

36.4
65.7

32.5
64.1

33.3
66.1

33.0
66.1

35.0
69.6

31.8
68.3
...

!!!!!
"""


# Monty Hall Paradox 2


def montyhall2(num_doors=10, choice=5, k=1, change=False):
    doors = [i for i in range(1, num_doors + 1)]
    labels = ["Yes"] + ["No" for _ in range(num_doors - 1)]
    np.random.shuffle(labels)

    choices = dict(zip(doors, labels))

    del_num = k
    while del_num > 0:
        keys = list(choices.keys())
        num = np.random.choice(keys)
        if num != choice and choices[num] != "Yes":
            del_num -= 1
            del choices[num]

    if change:
        del choices[choice]
        keys1 = list(choices.keys())
        key = np.random.choice(keys1)
        choices = {key: choices[key]}
    else:
        choices = {choice: choices[choice]}

    return choices


def call_mh2(ks, num_doors=10, choice=5, num_iter=100):
    n1s = []
    n2s = []

    for _ in range(num_iter):
        n1 = []
        n2 = []
        for k in ks:
            r1 = montyhall2(num_doors=num_doors, choice=choice, k=k, change=False)
            r2 = montyhall2(num_doors=num_doors, choice=choice, k=k, change=True)
            n1.extend(list(r1.values()))
            n2.extend(list(r2.values()))

        n1s.append(len([h for h in n1 if h == "Yes"]) / len(n1) * 100)
        n2s.append(len([h for h in n2 if h == "Yes"]) / len(n2) * 100)

    return n1s, n2s


ks = [1, 2]

n1s, n2s = call_mh2(ks=ks, num_doors=4, choice=1, num_iter=10)








