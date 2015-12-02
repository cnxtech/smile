import models
import submission
import plot_digits
import preprocessor
import os.path
from sklearn.externals import joblib
from sklearn.metrics import classification_report

def please():
	target_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	tr_identity, tr_labels, tr_images, valid_identity, valid_labels, valid_images = preprocessor.load_train_and_valid()
	test_images = preprocessor.load_test()
	unlabeled_images = preprocessor.load_unlabeled()
	
	pca_filename = 'pca.pkl'
	if os.path.isfile(pca_filename):
		pca = joblib.load(pca_filename)
	else: 
		# save the PCA
		pca = models.PCA()
		pca.fit(unlabeled_images)
		joblib.dump(pca, pca_filename)

	pca_tr_images = pca.transform(tr_images)
	pca_valid_images = pca.transform(valid_images)

	model = models.AdaBoost()
	model.fit(pca_tr_images, tr_labels)
	train_predictions = model.predict(pca_tr_images)
	valid_predictions = model.predict(pca_valid_images)

	print "########### Model " + model.getName() + " ##################\n"
	print "########### Valid ##################\n"
	print(classification_report(valid_labels, valid_predictions, target_names=target_names))
	print "\n########### Train ##################\n"
	print(classification_report(tr_labels, train_predictions, target_names=target_names))

	# submission.output(predictions)

if __name__ == '__main__':
	please() # work
