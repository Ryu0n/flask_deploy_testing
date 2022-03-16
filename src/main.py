import numpy as np
import tensorflow.keras as keras

from flask import Flask, request
from src.mnist import MNIST
from src.utils.util_path import PathUtils

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    message = """
    send params with test image index (1 ~ 10000)\n
    \n
    ex)\n
    {'index': 1}
    """
    return message


def predict(index):
    _, _, x_test, y_test = MNIST.instance().load_with_preprocess()
    x, y = x_test[index], y_test[index]
    y_pred = model.predict(x[np.newaxis, ...])  # softmax distribution (0부터 9까지 확률값)
    y_pred = np.argmax(y_pred)
    return f'label : {y}, prediction : {y_pred}'


@app.route('/predict_get')
def make_prediction_get():
    """
    GET 방식
    localhost:8000/predict_get?index=<int>
    index: 테스트를 수행할 이미지 데이터셋의 인덱스
    :return:
    """
    parameters = request.args.to_dict()
    index = parameters['index']
    return predict(int(index))


@app.route('/predict', methods=['POST'])
def make_prediction():
    """
    POST 방식
    localhost:8000/predict

    params
    - index: int : 테스트를 수행할 이미지 데이터셋의 인덱스
    :return:
    """
    if request.method == 'POST':
        params = request.get_json()
        index = params['index']
        return predict(int(index))


if __name__ == '__main__':
    model = keras.models.load_model(PathUtils.model_path() + 'demo_classifier.h5')
    app.run(host='localhost', port=8000, debug=True)
