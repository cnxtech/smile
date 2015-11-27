import csv
from conf import paths

def output(predictions):
    with open(paths.OUTPUT, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Prediction'])

        samples = predictions.shape[0]
        for i in xrange(samples):
            writer.writerow([i + 1, predictions[i]])

        if samples < 1253:
            for i in xrange(1253 - samples):
                writer.writerow([samples + i + 1, 0])
