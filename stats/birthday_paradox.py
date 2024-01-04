import matplotlib.pyplot as plt
import numpy as np





# Birthday Paradox 1

def match_birthday(birthday, num_iter=50):

    num_simulate = []  # how many iters to find the match
    num = birthday["M"] * birthday["D"]  # find the day number in the year

    for i in range(num_iter):
        count = 0
        match = False
        while not match:
            run = np.random.choice([i for i in range(1, 366)])
            if run != num:
                count += 1
            else:
                count += 1
                match = True
                num_simulate.append(count)

    return num_simulate


num_simulate = match_birthday(birthday={"M": 1, "D": 5}, num_iter=200)

plt.hist(num_simulate, bins=len(num_simulate))
plt.title("Birthday Paradox 1")
plt.xlabel("Iteration")
plt.ylabel("Match")
plt.show()


plt.scatter([i for i in range(len(num_simulate))], num_simulate)
plt.title("Birthday Paradox 1")
plt.xlabel("Try")
plt.ylabel("Match Found")
plt.show()







# Birthday Paradox 2

def simulator1(num_st=365, num_iter=1000):

    birthdays = [np.random.choice([d for d in range(1, 366)]) for _ in range(num_st)]
    match = 0

    for _ in range(num_iter):
        num = np.random.choice([i for i in range(1, 366)])
        if num in birthdays:
            match += 1

    return match / num_iter


num_st = [i*10 for i in range(1, 101)]
matches = []

for num in num_st:
    match = simulator1(num_st=num)
    matches.append(match)

plt.hist(matches, bins=len(matches))
plt.title("Birthday Paradox 2")
plt.show()








# Birthday Paradox 3

def simulator2(num_st, num_iter=100):

    match = 0
    for _ in range(num_iter):
        birthdays = [np.random.choice([d for d in range(1, 366)]) for _ in range(num_st)]
        birthday = np.random.choice([i for i in range(len(birthdays))])
        if birthday in birthdays:
            match += 1

    return match / num_iter


num_st1 = [i*20 for i in range(1, 51)]
matches1 = []

for num in num_st1:
    match = simulator2(num_st=num)
    matches1.append(match)

plt.scatter(num_st1, matches1, label="Probabilities")
plt.plot([0.5 for _ in range(1000)], color="red", label="P 50")
plt.title("Probability vs. Num St")
plt.xlabel("Nim St")
plt.ylabel("Probability")
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.show()








# Birthday Paradox 4

def simulator3(num_st, num_iter=100):

    matches = []
    for num in num_st:

        match = 0
        for _ in range(num_iter):

            birthdays = [np.random.choice([d for d in range(1, 366)]) for _ in range(num)]
            uniq_birthdays = set(birthdays)

            diff = len(birthdays) - len(list(uniq_birthdays))
            match += diff

        matches.append(match)

    return matches


num_st4 = [i*20 for i in range(50)]

matches4 = simulator3(num_st4)


plt.scatter(num_st4, matches4, label="Probabilities")

plt.title("Frequency vs. Num St")
plt.xlabel("Num St")
plt.ylabel("Frequency")
plt.grid(True)
plt.legend()
plt.show()





