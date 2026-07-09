import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort


# CONFIGURAÇÕES — ajuste aqui se necessário

CAMERA_INDEX      = 1      
CONFIANCA_MINIMA  = 0.25     # controla o equilibrio entre detectar e nao detectar, dependendo do ambiente
TEMPO_SAIDA       = 3.0     # segundos sem aparecer antes de considerar saída
LARGURA_FRAME     = 640     # reduzido para melhor performance
ALTURA_FRAME      = 480     # reduzido para melhor performance
PROCESSAR_A_CADA  = 2       # roda YOLO a cada N frames 

# INICIALIZAÇÃO
model   = YOLO("yolov8x.pt")    
model.to('mps')
tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.6)
frame_count = 0
ultimo_tracks = []

video = cv2.VideoCapture(CAMERA_INDEX)
video.set(cv2.CAP_PROP_FRAME_WIDTH,  LARGURA_FRAME)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_FRAME)

# ESTADO
entrada      = {}   # {id: timestamp de entrada}
ultimo_visto = {}   # {id: timestamp ultima detecção}
historico    = {}   # {id: tempo total de permanência}
ativos       = set()

# CORES
COR_BOX      = (0, 255, 120)
COR_ID       = (255, 255, 255)
COR_PAINEL   = (20, 20, 20)
COR_TITULO   = (0, 200, 255)
COR_METRICA  = (200, 200, 200)
COR_DESTAQUE = (0, 255, 120)


# Converte segundos para string mm:ss
def formatar_tempo(segundos):
    
    m = int(segundos) // 60
    s = int(segundos) % 60
    return f"{m:02d}:{s:02d}"

# Desenhando o painel lateral com as métricas
def desenhar_painel(frame, pessoas_agora, historico):
    
    h, w = frame.shape[:2]
    largura_painel = 180

    # fundo semi-transparente do painel
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - largura_painel, 0), (w, h), COR_PAINEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    x = w - largura_painel + 14
    y = 36

    # título
    cv2.putText(frame, "MONITORAMENTO", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COR_TITULO, 2)
    y += 20
    cv2.line(frame, (x, y), (w - 14, y), COR_TITULO, 1)
    y += 24

    # pessoas agora
    cv2.putText(frame, "No ambiente agora:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_METRICA, 1)
    y += 28
    cv2.putText(frame, str(len(pessoas_agora)), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COR_DESTAQUE, 3)
    y += 44

    # total histórico
    total = len(historico)
    cv2.putText(frame, f"Total detectado: {total}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_METRICA, 1)
    y += 28

    # métricas de tempo
    tempos = list(historico.values())
    if tempos:
        media  = sum(tempos) / len(tempos)
        minimo = min(tempos)
        maximo = max(tempos)
        cv2.putText(frame, f"Tempo medio:  {formatar_tempo(media)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COR_METRICA, 1)
        y += 22
        cv2.putText(frame, f"Menor tempo:  {formatar_tempo(minimo)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COR_METRICA, 1)
        y += 22
        cv2.putText(frame, f"Maior tempo:  {formatar_tempo(maximo)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COR_METRICA, 1)
        y += 32

    # linha divisória
    cv2.line(frame, (x, y), (w - 14, y), (60, 60, 60), 1)
    y += 18

    # lista de IDs ativos com cronômetro
    cv2.putText(frame, "IDs no frame:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_METRICA, 1)
    y += 22

    agora = time.time()
    for pid in sorted(pessoas_agora):
        tempo_atual = agora - entrada.get(pid, agora)
        linha = f"  ID {int(pid):>3}  ->  {formatar_tempo(tempo_atual)}"
        cv2.putText(frame, linha, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COR_DESTAQUE, 1)
        y += 20
        if y > h - 20:
            break


# LOOP PRINCIPAL - captura dos frames da camera
print("Iniciando... pressione Q para sair.")

while True:
    success, frame = video.read()
    if not success:
        print("Erro ao acessar câmera.") # caso nao consiga acessar a camera
        break

    agora = time.time()
    frame_count += 1

    # detecção YOLO apenas a cada uma quantidade N de frames 
    if frame_count % PROCESSAR_A_CADA == 0:
        resultados = model(frame, classes=[0], conf=CONFIANCA_MINIMA,
                           verbose=False, imgsz=480, iou=0.45, max_det=50)
        deteccoes = []
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                deteccoes.append([x1, y1, x2, y2, conf])

        if len(deteccoes) > 0:
            tracks = tracker.update(np.array(deteccoes))
        else:
            tracks = tracker.update(np.empty((0, 5)))
        ultimo_tracks = tracks
    else:
        tracks = ultimo_tracks

    ids_no_frame = set()

    for track in tracks:
        x1, y1, x2, y2, pid = map(int, track)
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)

        # filtra detecções improváveis de ser humano - um teste
        largura = x2 - x1
        altura  = y2 - y1
        proporcao = altura / (largura + 1e-5)

        # se muito pequeno
        if largura < 30 or altura < 60:       
            continue
        # se proporção estranha
        if proporcao < 1.0 or proporcao > 3.5:  
            continue

        ids_no_frame.add(pid)
        ultimo_visto[pid] = agora

        # se nova pessoa detectada
        if pid not in entrada:
            entrada[pid] = agora
            ativos.add(pid)

        # tempo atual dessa pessoa
        tempo_atual = agora - entrada[pid]

        # caixa de detecção (bounding box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COR_BOX, 2)

        # informação na caixa com ID e cronômetro
        label = f"ID {pid} | {formatar_tempo(tempo_atual)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), COR_BOX, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # verificar quem saiu do frame
    ids_para_remover = []
    for pid in list(ativos):
        if pid not in ids_no_frame:
            tempo_sem_ver = agora - ultimo_visto.get(pid, agora)
            if tempo_sem_ver > TEMPO_SAIDA:
                # registra no histórico e remove dos ativos
                tempo_total = ultimo_visto[pid] - entrada[pid]
                if tempo_total > 0:
                    historico[pid] = tempo_total
                ids_para_remover.append(pid)

    for pid in ids_para_remover:
        ativos.discard(pid)

    # painel lateral com informações
    desenhar_painel(frame, ids_no_frame, historico)

    # exibindo
    frame_exibicao = cv2.resize(frame, (1280, 720))
    cv2.imshow("Contador de Pessoas | Permanencia", frame_exibicao)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# encerramento
video.release()
cv2.destroyAllWindows()

print("\n Resumo da sessão ")
print(f"Total de pessoas detectadas: {len(historico) + len(ativos)}")
if historico:
    tempos = list(historico.values())
    print(f"Tempo médio de permanência: {formatar_tempo(sum(tempos)/len(tempos))}")
    print(f"Menor permanência: {formatar_tempo(min(tempos))}")
    print(f"Maior permanência: {formatar_tempo(max(tempos))}")