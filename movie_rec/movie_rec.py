import json
from optparse import OptionParser

from lib.moviereviews import MovieReviews
from lib.mlhelper import create_inputs_matrix, create_target_matrix
from lib.multi_class import one_vs_all, predict_all

def load_data(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f]

if __name__ == '__main__':
    # parse options. inputs <-
    parser = OptionParser()
    parser.add_option("-i", "--inputs", dest="inputs", metavar="PATH",
                      help="inputs for X")
    parser.add_option("-p", "--positive", dest="positive", metavar="PATH",
                      help="examples of positive target values")
    parser.add_option("-n", "--negative", dest="negative", metavar="PATH",
                      help="examples of negative target values")
    (options, args) = parser.parse_args()   
   
    # load data 
    movie_reviews = MovieReviews(
        options.inputs,
        options.positive,
        options.negative
    )

    # ML formatted data <- data 
    X = create_inputs_matrix(
        movie_reviews.seen_movies,
        movie_reviews.critics,
        movie_reviews.critic_ratings
    )
    Y = create_target_matrix(
        movie_reviews.seen_movies,
        movie_reviews.liked_movies,
        movie_reviews.disliked_movies
    )

    # output <- params, ML formatted data
    learning_rate = 1
    num_classifications = 2
    all_theta = one_vs_all(X, Y, num_classifications, learning_rate)
    hypothesis = predict_all(X, all_theta)

    # output
    print hypothesis
    
