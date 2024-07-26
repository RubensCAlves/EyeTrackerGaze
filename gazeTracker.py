from __future__ import division
import os
import cv2
import dlib
from eye import Eye
from calibraction import Calibration


class GazeTracking(object):
    """
    Esta classe rastreia o olhar do usuário.
    Ela fornece informações úteis como a posição dos olhos
        e pupilas e permite saber se os olhos estão abertos ou fechados
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector é usado para detectar rostos
        self.face_detector = dlib.get_frontal_face_detector()
        #self.face_detector = dlib.get_frontal_face_detector()

        # _predictor é usado para obter marcos faciais de um determinado rosto
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """
        Verifique se a pupilas foi localizada
        """
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detecta o rosto e inicializa Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Atualiza o quadro e o analisa.

        Argumentos:
        frame (numpy.ndarray): O quadro a ser analisado
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Retorna as coordenadas da pupila esquerda"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Retorna as coordenadas da pupila direita"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """
        Retorna um número entre 0,0 e 1,0 que indica a
            direção horizontal do olhar. A extrema direita é 0,0,
            o centro é 0,5 e a extrema esquerda é 1,0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """
        Retorna um número entre 0,0 e 1,0 que indica a
            direção vertical do olhar. O topo extremo é 0,0,
            o centro é 0,5 e o fundo extremo é 1,0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Retorna verdadeiro se o usuário estiver olhando para a direita"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Retorna verdadeiro se o usuário estiver olhando para a esquerda"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Retorna verdadeiro se o usuário estiver olhando para o centro"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Retorna verdadeiro se o usuário fechar os olhos"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Retorna o quadro principal com as pupilas destacadas"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
