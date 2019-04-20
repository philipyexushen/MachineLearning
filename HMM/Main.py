import numpy as np
import random
from collections import Counter

def main():
    a0 = [0, 1, 2, 3, 4]
    a1 = [1, 2, 3, 4, 5]
    a2 = [0, 2, 3, 4, 5]
    a3 = [0, 1, 3, 4, 5]
    a4 = [0, 1, 2, 4, 5]
    a5 = [0, 1, 2, 3, 5]

    choice_sets = [a0, a1, a2, a3, a4, a5]

    p_each = 1.0 / 6.0
    A = np.array([  [p_each, p_each, p_each, p_each, p_each, p_each],
                    [p_each, p_each, p_each, p_each, p_each, p_each],
                    [p_each, p_each, p_each, p_each, p_each, p_each],
                    [p_each, p_each, p_each, p_each, p_each, p_each],
                    [p_each, p_each, p_each, p_each, p_each, p_each],
                    [p_each, p_each, p_each, p_each, p_each, p_each]], dtype=np.float)

    result = []
    this_index = random.randint(0, 5)
    this_set = choice_sets[this_index]
    for _ in range(10):
        result.append(random.choice(this_set))
        this_index = np.random.choice(6, 1, p=A[this_index])[0]
        this_set = choice_sets[this_index]

    word_counts = Counter(result)
    top_one = word_counts.most_common(1)[0]
    print(f"Most number={top_one[0]}, count={top_one[1]}")


if __name__ == "__main__":
    main()
