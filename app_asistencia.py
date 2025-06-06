import cv2
import dlib
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime

# --- Manejo de rutas para PyInstaller y modo script ---
if getattr(sys, 'frozen', False):
    # Cuando se ejecuta como exe con PyInstaller
    base_path = sys._MEIPASS         # Carpeta temporal donde PyInstaller extrae recursos
    exe_dir = os.path.dirname(sys.executable)  # Carpeta donde está el exe
else:
    # Cuando se ejecuta como script normal
    base_path = os.path.abspath(".")
    exe_dir = os.path.abspath(".")

def resource_path(relative_path):
    return os.path.join(base_path, relative_path)

# Rutas absolutas para archivos y carpetas usados
predictor_path = resource_path('shape_predictor_68_face_landmarks.dat')
face_rec_model_path = resource_path('dlib_face_recognition_resnet_model_v1.dat')
csv_alumnos_path = resource_path('alumnos.csv')
rostros_folder = resource_path('rostros')

# Ruta absoluta para guardar el CSV de asistencias junto al exe
csv_path = os.path.join(exe_dir, 'asistencias.csv')

# --- Inicialización modelos ---
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# --- Cargar alumnos ---
df_alumnos = pd.read_csv(csv_alumnos_path)  # columnas: id,nombre,imagen

def cargar_rostros_y_embeddings(carpeta_rostros=rostros_folder):
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

# --- Registrar asistencia ---
def registrar_asistencia(info):
    ahora = datetime.now()
    fecha = ahora.strftime('%Y-%m-%d')
    hora = ahora.strftime('%H:%M:%S')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['id', 'nombre', 'fecha', 'hora', 'imagen'])

    registrado = (
        (df['id'] == info['id']) & (df['fecha'] == fecha)
    ).any()

    if not registrado:
        nueva_fila = {
            'id': info['id'],
            'nombre': info['nombre'],
            'fecha': fecha,
            'hora': hora,
            'imagen': info['imagen']
        }
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"Asistencia registrada para {info['nombre']} ({info['id']})")
    else:
        print(f"{info['nombre']} ya estaba registrado hoy.")

# --- Reconocimiento facial ---
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
            info = metadatos_registrados[idx]
            nombre = info['nombre']
            registrar_asistencia(info)
        else:
            nombre = "Desconocido"

        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, nombre, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return frame

# --- Captura webcam ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

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
