import utils
import knn
import csv
from conf import paths

def please():
    tr_identity, tr_labels, tr_images = utils.load_train()
    test_images = utils.load_test()

    model = knn.KNeighbors(5)
    model.fit(tr_images, tr_labels)
    test_labels = model.predict(test_images)

    with open(paths.OUTPUT, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Prediction'])

        samples = test_labels.shape[0]
        for i in xrange(samples):
            writer.writerow([i + 1, test_labels[i]])

        if samples < 1253:
            for i in xrange(1253 - samples):
                writer.writerow([samples + i + 1, 0])

if __name__ == '__main__':
    please() # work
