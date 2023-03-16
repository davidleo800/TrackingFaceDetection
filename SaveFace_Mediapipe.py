import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Inicializar el detector facial
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se puede recibir imagen desde la cámara.")
            break

        # Convertir la imagen a BGR para su procesamiento por OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detectar caras en la imagen
        results = detector.process(image)

        # Dibujar los resultados en la imagen original
        if results.detections:
            for detection in results.detections:
                # Obtener las coordenadas de la cara
                bbox = detection.location_data.relative_bounding_box
                h, w, c = image.shape

                # Calcular las coordenadas absolutas de la cara
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Recortar la cara de la imagen original
                face_image = image[y:y+height, x:x+width]

                # Guardar la imagen recortada en un archivo
                cv2.imwrite("rostro.png", face_image)

                # Dibujar un rectángulo alrededor de la cara en la imagen original
                mp_drawing.draw_detection(image, detection)

        # Mostrar la imagen procesada
        cv2.imshow('Reconocimiento facial', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()