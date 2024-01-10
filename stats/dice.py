import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# rolling dice simulation


class DiceSimulator(object):

    def __init__(self, num_side=6, num_dice=1, probs=None):

        self.num_side = num_side
        self.sides = [i+1 for i in range(num_side)]
        self.num_dice = num_dice
        if not probs:
            self.probs = self.calc_prob(self.num_side)

        else:
            self.probs = probs

    def calc_prob(self, num_side):

        ps = [1 / num_side for _ in range(num_side)]

        probs = {self.sides[i]: ps[i] for i in range(num_side)}

        return probs


    def roll_dice(self):

        num = np.random.choice(self.sides)

        return num

    def roll_dice1(self, num_roll):

        rolls = []
        for _ in range(num_roll):

            roll = self.roll_dice()
            rolls.append(roll)

        return rolls

    def prob_(self, rolls, num, prob):

        prob1 = (rolls.count(num) * prob) / len(rolls)

        return prob1

    def call_prob_(self, rolls):

        nums = list(set(rolls))
        probs = {}

        for i in range(len(nums)):

            prob = self.prob_(rolls, nums[i], self.probs[nums[i]])
            probs[nums[i]] = prob

        return probs

    def dice(self, num_iter=100):

        result = {}

        for _ in range(self.num_dice):

            rolls = self.roll_dice1(num_roll=num_iter)
            probs = self.call_prob_(rolls)

            for key, val in probs.items():
                if key in result:
                    result[key] *= val
                else:
                    result[key] = val

        for key1, val1 in result.items():
            result[key1] = val1 * num_iter

        return result


dice_simulator = DiceSimulator(num_side=6, num_dice=1)

result = dice_simulator.dice(num_iter=1000)

print(result)
""" 
{
1: 28.666666666666664, 
2: 25.5, 
3: 29.666666666666664, 
4: 27.333333333333332, 
5: 24.666666666666664, 
6: 30.833333333333332
}
"""

dice_simulator = DiceSimulator(num_side=6, num_dice=2, probs={1: .2, 2: .3, 3: .1, 4: .25, 5: .05, 6: .1})

result = dice_simulator.dice(num_iter=1000)

print(result)
"""
{1: 1.1832, 2: 2.50965, 3: 0.24016000000000007, 4: 2.1359999999999997, 5: 0.05811, 6: 0.28302000000000005}

"""























