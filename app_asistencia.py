print("Cargando...")

import cv2
import dlib
import numpy as np
import os
import pandas as pd
from datetime import datetime
from collections import deque, defaultdict

# Configuración de archivos y modelos
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
archivo_alumnos = 'alumnos.csv'
archivo_asistencias = 'asistencias.csv'
carpeta_rostros = 'rostros'

# Inicializar modelos
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Cargar alumnos desde CSV
df_alumnos = pd.read_csv(archivo_alumnos)

# Estructuras para verificación de asistencia
ventana_confirmacion = deque(maxlen=10)
contador_confirmacion = defaultdict(int)
registrados_ya = set()

# Función para cargar rostros
def cargar_rostros_y_embeddings():
    descriptores = []
    metadatos = []

    for _, row in df_alumnos.iterrows():
        img_filename = row['imagen']
        img_path = os.path.join(carpeta_rostros, img_filename)

        if not os.path.exists(img_path):
            print(f"No se encontró la imagen {img_path}")
            continue

        img = dlib.load_rgb_image(img_path)
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = sp(img, dets[0])
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            descriptores.append(np.array(face_descriptor))
            metadatos.append({
                'id': row['id'],
                'nombre': row['nombre'],
                'imagen': row['imagen']
            })
        else:
            print(f"No se detectó rostro en {img_filename}")

    return metadatos, descriptores

metadatos_registrados, descriptores_registrados = cargar_rostros_y_embeddings()

# Función para registrar asistencia
def registrar_asistencia(info):
    ahora = datetime.now()
    fecha = ahora.strftime('%Y-%m-%d')
    hora = ahora.strftime('%H:%M:%S')

    if os.path.exists(archivo_asistencias):
        df = pd.read_csv(archivo_asistencias)
    else:
        df = pd.DataFrame(columns=['id', 'nombre', 'fecha', 'hora', 'imagen'])

    ya_registrado = (
        (df['id'] == info['id']) & (df['fecha'] == fecha)
    ).any()

    if not ya_registrado:
        nueva_fila = {
            'id': info['id'],
            'nombre': info['nombre'],
            'fecha': fecha,
            'hora': hora,
            'imagen': info['imagen']
        }
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
        df.to_csv(archivo_asistencias, index=False)
        print(f"Asistencia registrada para {info['nombre']} ({info['id']})")
    else:
        print(f"{info['nombre']} ya estaba registrado hoy.")

# Función principal de reconocimiento con confirmación
def reconocer_rostro(frame):
    global ventana_confirmacion, contador_confirmacion, registrados_ya

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 1)

    for d in dets:
        shape = sp(rgb, d)
        face_descriptor = facerec.compute_face_descriptor(rgb, shape)
        descriptor_actual = np.array(face_descriptor)

        distancias = [np.linalg.norm(descriptor_actual - desc) for desc in descriptores_registrados]
        idx = np.argmin(distancias)

        if distancias[idx] < 0.6:
            info = metadatos_registrados[idx]
            id_alumno = info['id']
            nombre = info['nombre']

            ventana_confirmacion.append(id_alumno)
            contador_confirmacion[id_alumno] += 1

            if ventana_confirmacion.count(id_alumno) >= 5 and id_alumno not in registrados_ya:
                registrar_asistencia(info)
                registrados_ya.add(id_alumno)
                ventana_confirmacion.clear()
                contador_confirmacion.clear()
        else:
            nombre = "Desconocido"

        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# Captura de video
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

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

if __name__ == "__main__":
    main()
