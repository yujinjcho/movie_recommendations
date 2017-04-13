import json
import numpy as np
from pprint import pprint

def load_data(file_name):
    with open(file_name) as f:
        return [json.loads(line.rstrip()) for line in f]

def create_rating_mapping(data):
    critics = set([review["critic"] for review in data])
    return dict([(critic, get_review(data, critic)) for critic in critics])

def get_review(data, critic):
    return dict([(review["movie"], int(review["liked"])) for review in data if critic == review["critic"]])

def create_x_matrix(data, review_mapping):
    return np.array([construct_x_row(movie, critics, review_mapping) for movie in movies])

def construct_x_row(movie, critics, review_mapping):
    x_row = [add_rating(critic, movie, review_mapping) for critic in critics]
    return [1] + x_row 

def add_rating(critic, movie, review_mapping):
    if critic in review_mapping and movie in review_mapping[critic]:
        return review_mapping[critic][movie]
    return 0

def load_test_set():
    return [
       	 1, # u'tangerine_2015',
	 0, # u'senna',
	 -1, # u'zoolander_2',
	 0, # u'a_monster_calls',
	 0, # u'manchester_by_the_sea',
	 1, # u'deadpool',
	 0, # u'land_of_mine',
	 0, # u'ghost_in_the_shell_2017',
	 0, # u'bloodrayne',
	 0, # u'graduation_2017',
	 0, # u'paterson',
	 0, # u'a_united_kingdom',
	 0, # u'logan_2017',
	 1, # u'split_2017',
	 1, # u'green_room_2016',
	 0, # u'neruda_2016',
	 0, # u'the_founder',
	 0, # u'alone_in_the_dark',
	 1, # u'machete',
	 0, # u'the_salesman_2017',
	 0, # u'gold_2017',
	 0, # u'assassins_creed',
	 0, # u'fences_2016',
	 0, # u'silence_2017',
	 -1, # u'popstart_never_stop_never_stopping',
	 0, # u'kong_skull_island',
	 0, # u'raw_2017',
	 1, # u'x_men_first_class',
	 0, # u'sabans_power_rangers',
	 0, # u'beauty_and_the_beast_2017',
	 0, # u'dark_night_2017' 
    ]

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

def predict(x, theta, movies, y):
    seen_movies = [movie[0] for movie in zip(movies, y) if movie[1] != 0]
    scores = np.sum(np.multiply(x, theta), axis=1)
    sorted_scores = sorted(zip(movies, scores), key=lambda x: x[1], reverse=True)
    for rec in sorted_scores:
        if rec[0] not in seen_movies:
            print rec[0], rec[1] 

data = load_data('sample.json')
review_mappings = create_rating_mapping(data)
movies = list(set([review["movie"] for review in data]))
critics = list(set([review["critic"] for review in data]))

x = create_x_matrix(data, review_mappings)
y = load_test_set()
m, n = np.shape(x)
numIterations = 10000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
predictions = predict(x, theta, movies, y)

