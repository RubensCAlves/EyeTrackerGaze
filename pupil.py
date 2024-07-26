import numpy as np
import cv2


class Pupil(object):
    """
    Esta classe detecta a íris de um olho e estima
        a posição da pupila
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Executa operações no quadro do olho para isolar a íris

            Argumentos:
            eye_frame (numpy.ndarray): Quadro contendo um olho e nada mais
            threshold (int): Valor limite usado para binarizar o quadro do olho

            Retorna:
            Um quadro com um único elemento representando a íris
            cv.Stitcher_PANORAMA
        """
        kernel = np.ones((3, 3))
        eye_frame = cv2.bilateralFilter(eye_frame, 0, 75, 75)
        eye_frame = cv2.erode(eye_frame, kernel, iterations=3)
        eye_frame = cv2.threshold(eye_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        return eye_frame

    def detect_iris(self, eye_frame):
        """Detecta a íris e estima a posição da íris
            calculando o centroide.

            Argumentos:
            eye_frame (numpy.ndarray): Quadro contendo um olho e nada mais
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass
