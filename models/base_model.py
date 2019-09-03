import abc
import logging
import pickle

from os.path import join
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BaseModel(metaclass=abc.ABCMeta):
    """Sklearn models meta class"""

    def __init__(self):
        self.name = None
        self.model = None
        self.best_model = None

    @property
    def model(self):
        """Return model attribute
        :return: model
        :rtype: sklearn.linear_model.base.LinearModel
        """

        if hasattr(self, '_model'):
            return self._model

    @model.setter
    def model(self, value):
        """model attribute setter
        :param value: sklearn regression model
        :type value: sklearn.linear_model.base.LinearModel
        """

        if not hasattr(self, '_model'):
            self._model = value
        else:
            if self._model is None:
                self._model = value
            else:
                logging.info('model field is not empty')

    @property
    def best_model(self):
        """Return the best model attribute
        :return: model
        :rtype: sklearn.linear_model.base.LinearModel
        """

        if hasattr(self, '_best_model'):
            return self._best_model

    @best_model.setter
    def best_model(self, value):
        """best_model attribute setter
        :param value: sklearn regression model
        :type value: sklearn.linear_model.base.LinearModel
        """

        if not hasattr(self, '_best_model'):
            self._best_model = value
        else:
            if self._best_model is None:
                self._best_model = value
            else:
                logging.info('model field is not empty')

    def train(self, x, y):
        """Model training method
        :param x: input values
        :type x: numpy.ndarray
        :param y: input values
        :type y: numpy.ndarray
        """

        tuned_parameters = self.get_params()

        model = GridSearchCV(
            self.model, tuned_parameters, cv=5, scoring='accuracy', refit=True
        )
        model.fit(x, y)

        self.best_model = model.best_estimator_

    def predict(self, x):
        """Predict the value of x using the best trained model
        :param x: input values
        :type x: np.array
        :return: predicted values
        :rtype: np.array
        """

        return self.best_model.predict(X=x)

    @abc.abstractmethod
    def get_params(self):
        """An abstract method returning training parametes

        :return: model parameters
        :rtype: dict
        """

        pass

    def save_model(self, save_dir):
        """Model saving method
        :param save_dir: saving directory
        :type save_dir: str
        """

        file_name = self.__class__.__name__ if self.name is None else self.name
        file_name = '{}.pkl'.format(file_name)
        save_path = join(save_dir, file_name)

        model = self.model if self.best_model is None else self.best_model

        with open(save_path, 'wb') as file:
            pickle.dump(model, file)

        return True

    def load_model(self, model_dir):
        """Model loading method
        :param model_dir: saving directory
        :type model_dir: str
        """

        file_name = self.__class__.__name__ if self.name is None else self.name
        file_name = '{}.pkl'.format(file_name)
        load_path = join(model_dir, file_name)

        with open(load_path, 'rb') as file:
            self.best_model = pickle.load(file)

        return True