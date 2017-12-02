import cv2


class StatModel(object):
    def load(self, fn):
        self.model = cv2.ml.SVM_load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=10, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()


def evaluate(model, test_data, test_label, test_image):
    response = model.predict(test_data)
    error = (response != test_label)
    print(error)
    print('Accuracy: %.2f' % ((1 - error.mean()) * 100))
    print('predict: ', response[error])
    print('answer: ', test_label[error])
    for image in test_image[error]:
        cv2.imshow('figure', image)
        cv2.waitKey(0)
