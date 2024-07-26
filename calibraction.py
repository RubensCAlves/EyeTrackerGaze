#from __future__ import division
#import cv2
from pupil import Pupil


class Calibration(object):
    """
    Esta classe calibra o algoritmo de detecção de pupila encontrando o
        melhor valor limite de binarização para a pessoa e a webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames
    
    def threshold(self, side):
        """Retorna o valor limite para o olho determinado.

            Argumento:
            lado: Indica se é o olho esquerdo (0) ou o olho direito (1)
        """
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """Retorna a porcentagem de espaço que a íris ocupa
            a superfície do olho.
            Argumento:
            frame (numpy.ndarray): quadro de íris binarizado
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - frame

        #nb_blacks = nb_pixels - cv2.countNonZero(frame)

        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calcula o limite ideal para binarizar o
            moldura para o olho determinado.

            Argumento:
            eye_frame (numpy.ndarray): Quadro do olho a ser analisado
        """
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        #best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        best_threshold, average_iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Melhora a calibração levando em consideração o dada imagem.

            Argumentos:
            eye_frame (numpy.ndarray): Moldura do olho
            lado: Indica se é o olho esquerdo (0) ou o olho direito (1)
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)