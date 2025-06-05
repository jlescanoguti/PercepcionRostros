import cv2
import dlib
import numpy as np
import os
import sys

# Función para rutas relativas que funciona con PyInstaller
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Cargar modelos con rutas absolutas (funciona con PyInstaller)
predictor_path = resource_path('shape_predictor_68_face_landmarks.dat')
face_rec_model_path = resource_path('dlib_face_recognition_resnet_model_v1.dat')

print("Intentando cargar modelos...")

if not os.path.isfile(predictor_path) or not os.path.isfile(face_rec_model_path):
    print(f"Error: no se encontraron los archivos de modelo necesarios en {os.path.abspath('.')}")
    sys.exit(1)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

print("Modelo de shape predictor cargado OK")
print("Modelo de reconocimiento facial cargado OK")

# Ruta carpeta rostros, con resource_path para PyInstaller
carpeta_rostros = resource_path('rostros')

print(f"Carpeta rostros existe?: {os.path.isdir(carpeta_rostros)}")
print(f"Contenido carpeta rostros: {os.listdir(carpeta_rostros) if os.path.isdir(carpeta_rostros) else 'No existe'}")

def cargar_rostros_y_embeddings(carpeta_rostros=carpeta_rostros):
    nombres = []
    descriptores = []

    print(f"Cargando imágenes desde: {carpeta_rostros}")
    for archivo in os.listdir(carpeta_rostros):
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            img_path = os.path.join(carpeta_rostros, archivo)
            print(f"Cargando imagen: {img_path}")
            img = dlib.load_rgb_image(img_path)

            dets = detector(img, 1)
            if len(dets) > 0:
                shape = sp(img, dets[0])
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                descriptores.append(np.array(face_descriptor))
                nombre = os.path.splitext(archivo)[0]
                nombres.append(nombre)
            else:
                print(f"No se detectó rostro en {archivo}")
    return nombres, descriptores

nombres_registrados, descriptores_registrados = cargar_rostros_y_embeddings()

def reconocer_rostro(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 1)

    for d in dets:
        shape = sp(rgb, d)
        face_descriptor = facerec.compute_face_descriptor(rgb, shape)
        descriptor_actual = np.array(face_descriptor)

        distancias = [np.linalg.norm(descriptor_actual - desc) for desc in descriptores_registrados]
        idx = np.argmin(distancias)

        if distancias[idx] < 0.6:
            nombre = nombres_registrados[idx]
        else:
            nombre = "Desconocido"

        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, nombre, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_procesado = reconocer_rostro(frame)
    cv2.imshow("Reconocimiento Facial", frame_procesado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



