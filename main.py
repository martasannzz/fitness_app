# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 20:42:43 2025

@author: marta
"""

from ultralytics import YOLO
import cv2
import pandas as pd
from feedback_rules import exercise_functions, phase_detectors
from dotenv import load_dotenv
import openai
import os

def analizar_video(video_path, ejercicio, usar_gpt=True):
    # -------------------------------
    # ✅ 0. Cargar variables de entorno y API key
    # -------------------------------
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # -------------------------------
    # ✅ 1. Definir prompt del sistema (entrenador GPT)
    # -------------------------------
    SYSTEM_PROMPT = """Eres un entrenador personal experto en biomecánica y técnica de ejercicios. Tu estilo es claro, cercano y motivador, como si estuvieras entrenando a alguien en el gimnasio.
    Tu objetivo es ayudar a mejorar la técnica de forma práctica, honesta y útil, sin buscar errores donde no los hay. No hagas un analisis por fase o repetición si no es necesario.
    
    Sigue estas pautas:
    - Si la técnica es correcta, reconócelo claramente. No inventes fallos si no los hay.
    - Si hay errores, identifica solo los importantes. Evita ser excesivamente crítico.
    - Explica por qué un punto es bueno o malo, y cómo corregirlo de forma sencilla si es necesario.
    - No repitas fases o repeticiones de forma mecánica. Resume lo importante como lo harías hablando con alguien.
    - Si hay cosas que se pueden mejorar pero no son graves, trátalas como recomendaciones suaves, no como fallos.
    
    Finaliza con un resumen general si hay patrones, y con consejos realistas para seguir progresando.
    
    No uses tecnicismos innecesarios ni des diagnósticos clínicos. No inventes datos si no están disponibles.
    """
    
    # -------------------------------
    # ✅ 2. Cargar modelo YOLOv8-Pose
    # -------------------------------
    model = YOLO('yolov8n-pose.pt')
    
    # ✅ 3. Ruta del video de entrada
    #video_path = r"C:/Users/marta/Desktop/MASTER/TFM/input/archive/verified_data/verified_data/data_btc_10s/deadlift/3e5f9de9-7dc3-4149-889b-316336eab88e.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # ✅ 4. VideoWriter para guardar salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = cv2.VideoWriter('output_feedback.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    if not cap.isOpened():
        print("❌ Error: no se pudo abrir el vídeo.")
    else:
        print("✅ Vídeo cargado correctamente.")
    
    # ✅ 5. Ejercicio a analizar
    ejercicio = 'deadlift'  # Cambiar a squat, biceps_curl, bench_press según corresponda
    
    # 🔄 Inicializar variables
    feedback_storage = []
    previous_phase = None
    current_repetition = 0
    
    # ✅ Fases esperadas por ejercicio
    expected_phases_map = {
        "squat": {"setup", "eccentric", "bottom", "concentric"},
        "deadlift": {"setup", "pull", "lockout", "lowering"},
        "bench_press": {"setup", "lowering", "press"},
        "biceps_curl": {"setup", "curl", "lowering"}
    }
    
    # ✅ Secuencias válidas de fases para detectar una repetición (en orden)
    valid_phase_sequences = {
        "deadlift": ["pull", "lockout", "lowering", "setup"],
        "squat": ["eccentric", "bottom", "concentric", "setup"],
        "bench_press": ["lowering", "press", "setup"],
        "biceps_curl": ["curl", "lowering", "setup"]
    }
    
    expected_phases = expected_phases_map.get(ejercicio, set())
    valid_sequence = valid_phase_sequences.get(ejercicio, [])
    
    # Seguimiento de fases detectadas
    rep_phase_sequence = []
    last_counted_phase = None
    
    # ✅ Procesamiento frame a frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        results = model(frame)
    
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints_all = results[0].keypoints.xy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx_principal = areas.argmax()
            keypoints = keypoints_all[idx_principal]
    
            if ejercicio in phase_detectors:
                phase_info = phase_detectors[ejercicio](keypoints)
                phase = phase_info[0]
                other_angles = phase_info[1:]
                if ejercicio in exercise_functions:
                    angle, feedback = exercise_functions[ejercicio](keypoints, phase)
                else:
                    angle, feedback = None, "Ejercicio no implementado aún."
            else:
                phase = "unknown"
                angle, feedback = None, "Ejercicio no detectado."
    
            # ✅ Recuento por secuencia dinámica de fases
            if phase != "unknown" and phase != last_counted_phase:
                last_counted_phase = phase
    
                if not rep_phase_sequence or phase != rep_phase_sequence[-1]:
                    rep_phase_sequence.append(phase)
    
                if all(p in rep_phase_sequence for p in valid_sequence):
                    valid_indices = [rep_phase_sequence.index(p) for p in valid_sequence]
                    if valid_indices == sorted(valid_indices):
                        current_repetition += 1
                        print(f"✅ Repetición detectada: {current_repetition}")
                        rep_phase_sequence = []
    
            # ✅ Guardar feedback por fase
            exists_same_feedback = any(
            (entry['exercise'] == ejercicio) and
            (entry['phase'] == phase) and
            (entry['repetition'] == current_repetition) and  # usar current_repetition directamente
            (entry['feedback'] == feedback)
            for entry in feedback_storage
        )
        
        if not exists_same_feedback:
            feedback_storage.append({
                'exercise': ejercicio,
                'phase': phase,
                'repetition': current_repetition,  # igual aquí
                'feedback': feedback,
                'angle': angle
            })
    
            # ✅ Dibujar keypoints
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    
            label = f"{ejercicio.title()} Rep: {current_repetition} [{phase}]"
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        else:
            cv2.putText(frame, "No se detecta persona", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        output_video_path.write(frame)
        cv2.imshow('Exercise Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ✅ Liberar recursos
    cap.release()
    output_video_path.release()
    output_video_path.destroyAllWindows()
    
    # ✅ Convertir feedback a DataFrame
    feedback_df = pd.DataFrame(feedback_storage)
    print("✅ Feedback DataFrame generado:")
    print(feedback_df)
    
    # -------------------------------
    # ✅ Función GPT para una repetición individual
    # -------------------------------
    def obtener_feedback_gpt(angles_dict, ejercicio):
        prompt_usuario = f"Este es el resultado del análisis de keypoints de un {ejercicio}: {angles_dict}"
    
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=500
        )
        return response
    
    
    # -------------------------------
    # ✅ GPT Feedback del ejercicio completo
    # -------------------------------
    #usar_gpt = input("¿Deseas obtener feedback GPT del ejercicio completo? (s/n): ")
    
    if usar_gpt.lower() == 's':
        agrupado = (
            feedback_df.groupby(['repetition', 'phase'])
            .agg({
                'feedback': lambda x: ' '.join(set(x)),
                'angle': lambda x: [a for a in x if a is not None]
            })
            .reset_index()
        )
    
        num_reps = len(agrupado['repetition'].unique())
    
        if num_reps == 1:
            rep_text = "Este análisis corresponde a 1 repetición del ejercicio. Da feedback en singular, como si estuvieras observando un único intento/repetición/comprobación. No uses frases como 'a veces', 'en algunas repeticiones', 'en ciertos momentos', 'algún',etc. Ignora la variabilidad entre frames; resume el desempeño como una única ejecución. La repeticion ha sido dividida por fases, junto con datos de feedback y ángulos articulares."
        else:
            rep_text = f"Este análisis corresponde a {num_reps} repeticiones del ejercicio. Las repeticiones han sido divididaa por fases, junto con datos de feedback y ángulos articulares."
        
        agrupado['repetition'] = agrupado['repetition']
    
        resumen_agrupado = []
        for rep in sorted(agrupado['repetition'].unique()):
            rep_data = agrupado[agrupado['repetition'] == rep]
            resumen_agrupado.append({
                'repeticion': int(rep),
                'fases': rep_data['phase'].tolist(),
                'feedbacks': rep_data['feedback'].tolist(),
                'angulos': rep_data['angle'].tolist()
            })
    
        prompt_usuario = f"""
        Hola entrenador. Este es el análisis técnico del ejercicio "{ejercicio}". {rep_text}.
    
        Tu tarea es dar feedback como si fueras un entrenador experimentado observando el ejercicio completo. No me interesa un análisis por fases. Quiero que:
    
        - Expliques qué se está haciendo bien, si hay algo destacable.
        - Detectes los errores técnicos importantes, diciendo en qué fase ocurre si es relevante.
        - Expliques por qué importa corregirlo y cómo se puede mejorar, con consejos prácticos y claros.
    
        Ve al grano. No estructures por repetición ni por fase. No des tecnicismos ni explicaciones largas. El objetivo es que el usuario entienda qué está haciendo mal y cómo corregirlo para mejorar su técnica.
    
        Aquí tienes los datos:
        {resumen_agrupado}
        """
    
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=800
        )
    
        print("\n📋 Feedback técnico del ejercicio completo:")
        print(response['choices'][0]['message']['content'])
        
        return feedback_df, resumen_agrupado, output_video_path