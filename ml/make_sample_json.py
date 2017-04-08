import sys

import json

for line in sys.stdin:
    data = line.rstrip().split(",")
    critic = data[0]
    movie = data[1]
    liked = data[2]
    review = {
        'critic': critic,
        'movie': movie,
        'liked': liked
    }
    print json.dumps(review)
