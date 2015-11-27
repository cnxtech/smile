import knn
import submission
import plot_digits
import preprocessor
from sklearn.metrics import classification_report

def please():
    tr_identity, tr_labels, tr_images, valid_identity, valid_labels, valid_images = preprocessor.load_train_and_valid()
    test_images = preprocessor.load_test()

    model = knn.KNeighbors(5)
    model.fit(tr_images, tr_labels)
    predictions = model.predict(valid_images)

    target_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(classification_report(valid_labels, predictions, target_names=target_names))

    # submission.output(predictions)

if __name__ == '__main__':
    please() # work
