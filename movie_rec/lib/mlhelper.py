import numpy as np

def create_inputs_matrix(seen_movies, critics, review_mapping):
    return np.array([
        create_inputs_row(movie, critics, review_mapping) for movie in seen_movies
    ])

def create_inputs_row(movie, critics, review_mapping):
    return [add_rating(critic, movie, review_mapping) for critic in critics]

def add_rating(critic, movie, review_mapping):
    return review_mapping[critic].get(movie, 0)

def create_target_matrix(seen_movies, positives):
    positive_set = set(positives)
    return np.array([[1 if movie in positive_set else 2] for movie in seen_movies])

