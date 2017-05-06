import json
from optparse import OptionParser
from os import listdir
from collections import defaultdict
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f] 

def load_predictions(filename):
    with open(filename) as f:
        data = json.loads(f.read())
    classified = dict([
	(movie, '1') 
	if rating['like'] > rating['dislike']
	else (movie, '2')
	for movie, rating in data.items()
    ])
    return classified


def init_stats():
    stats = {}
    chart_type = ['accuracy', 'precision', 'recall', 'f-score']
    for accuracy_type in chart_type:
        stats[accuracy_type] = defaultdict(list)
    return stats

def evaluate_accuracy(predictions, lmbda, positives, negatives, stats):
    total = len(positives) + len(negatives)

    pos_correct = sum([ 1 if predictions[movie] == '1' else 0 for movie in positives])
    neg_correct = sum([ 1 if predictions[movie] == '2' else 0 for movie in negatives])
    correct = pos_correct + neg_correct
    pos_predicted = pos_correct + (len(negatives) - neg_correct)

    precision = pos_correct / float(pos_predicted)
    recall = pos_correct / float(len(positives))
    f_score = 2 * precision * recall / (precision + recall)

    stats['accuracy'][lmbda].append((correct / float(total)) * 100)
    stats['precision'][lmbda].append(precision * 100)
    stats['recall'][lmbda].append(recall * 100)
    stats['f-score'][lmbda].append(f_score * 100)

    print "Model accurately predicted {} / {}".format(correct, total)
    print "Accuracy: {0:.1f}%".format((correct / float(total)) * 100)
    print "Precision: {0:.1f}%".format(precision * 100)
    print "Recall: {0:.1f}%".format(recall * 100)
    print "F-Score: {0:.2f}".format(f_score)

def chart_stats(stats):
    for stat_type, stat_values in stats.items():
        chart_data = sorted(stat_values.items(), key=lambda x:float(x[0]))
        x_array = [score[0] for score in chart_data]
        y_array = [score[1] for score in chart_data]
        plt.plot(x_array, y_array, label=stat_type, alpha=0.5)

    plt.xlabel('Lambda')
    plt.ylabel('%')
    plt.title('Prediction Metrics')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # parse options. inputs <-
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", metavar="PATH",
                      help="path to predicted classes")
    parser.add_option("-p", "--positive", dest="positive", metavar="PATH",
                      help="path to positive test set")
    parser.add_option("-n", "--negative", dest="negative", metavar="PATH",
                      help="path to negative test set")
    (options, args) = parser.parse_args()   
   
    positives = load_data(options.positive)
    negatives = load_data(options.negative)

    stats = init_stats()
    for filename in listdir(options.input):
        predictions = load_predictions(options.input + '/' + filename)
        lmbda = filename.split('_')[1]
	print "\nFor lambda of", lmbda
        evaluate_accuracy(predictions, lmbda, positives, negatives, stats)

    chart_stats(stats)
