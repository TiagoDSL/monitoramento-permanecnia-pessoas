# 👁️ Sistema de Monitoramento de Permanência de Pessoas em Tempo Real

Sistema de visão computacional que detecta, rastreia e cronometra o tempo de permanência de pessoas em ambientes fechados — em tempo real, com apenas uma câmera comum.


Projeto desenvolvido como parte do portfólio em Ciência de Dados e Visão Computacional.




📸 Sistema em funcionamento

Mostrar Imagem

Sistema rodando em ambiente real: 4 pessoas detectadas simultaneamente com IDs únicos, cronômetros individuais e painel de métricas ao vivo.


💡 O Problema que esse projeto resolve

Estabelecimentos como lojas, restaurantes e clínicas não sabem quantas pessoas estão no ambiente em tempo real — e muito menos quanto tempo cada uma permanece. Coletar esse dado manualmente é inviável. Soluções comerciais existem, mas são caras e complexas.

Este sistema resolve isso com visão computacional e uma câmera comum:


Detecta automaticamente cada pessoa no ambiente
Atribui um identificador único e rastreia individualmente
Cronometra o tempo de permanência em tempo real
Gera métricas consolidadas ao vivo no painel


Onde isso pode ser aplicado

SetorAplicaçãoVarejoTempo médio do cliente por seção da lojaRestaurantesControle de rotatividade de mesasSaúdeMonitoramento de lotação em salas de esperaMuseusIdentificação de obras com maior engajamentoSegurançaAlerta de permanência em áreas restritas


🧠 O que esse projeto demonstra

Do ponto de vista técnico e analítico, o projeto envolve:


Pipeline de dados em tempo real — captura, processamento e exibição de informações frame a frame
Integração de modelos — combinação de detecção (YOLO) com rastreamento (SORT) em um fluxo coeso
Tomada de decisão baseada em dados — calibração iterativa de parâmetros com base em métricas observadas durante os testes
Tratamento de ruído — filtros para eliminar falsos positivos e estabilizar o rastreamento
Geração de métricas — cálculo de tempo médio, mínimo e máximo de permanência por sessão



🛠️ Stack Tecnológico

TecnologiaVersãoFunçãoPython3.10Linguagem principalYOLOv8 Extra Large8.4.xDetecção de pessoas em cada frameSORT—Rastreamento com filtro de KalmanOpenCV4.13Captura da câmera e processamento de framesNumPy2.2.6Manipulação de matrizes e dados numéricosPyTorch2.12.1Backend do YOLOv8

Decisões técnicas relevantes

Por que YOLOv8 Extra Large e não o modelo padrão?
O modelo nano apresentou dificuldades para detectar pessoas sentadas, de costas e longe da câmera — cenários comuns em ambientes reais. O modelo extra large resolveu esses casos com maior precisão em troca de maior custo computacional, compensado pelo uso do MPS (GPU Apple Silicon).

Por que SORT e não apenas YOLO?
Sem tracking, o sistema saberia apenas que há pessoas no frame — não quem é quem entre frames consecutivos. O SORT usa filtro de Kalman para prever a posição de cada pessoa no próximo frame e associa as detecções ao mesmo ID, tornando possível o cronômetro individual.


🏗️ Arquitetura do Pipeline

📷 Câmera (iPhone via Continuity Camera)
    ↓
🔍 YOLOv8x — detecta pessoas e retorna bounding boxes com score de confiança
    ↓
🔄 SORT — associa cada detecção a um ID único e estável entre frames
    ↓
⏱️ Lógica de permanência — registra entrada, calcula cronômetro, detecta saída
    ↓
📊 Painel OpenCV — exibe métricas atualizadas em tempo real


📊 Métricas geradas pelo sistema

O painel lateral exibe em tempo real:


Pessoas no ambiente agora — contagem ao vivo
Total histórico — quantas pessoas passaram desde o início da sessão
Tempo médio de permanência — média de todos os IDs encerrados
Menor tempo registrado — passagem mais rápida
Maior tempo registrado — permanência mais longa
Lista de IDs ativos — cada pessoa com seu cronômetro individual



⚙️ Instalação

Pré-requisitos


Python 3.10
Miniforge arm64 (obrigatório para Apple Silicon)
Git


1. Clone o repositório

bashgit clone https://github.com/seu-usuario/monitoramento-permanencia.git
cd monitoramento-permanencia

2. Crie e ative o ambiente

bashconda create -n visao_projeto python=3.10 -y
conda activate visao_projeto

3. Instale as dependências

bashpython -m pip install -r requirements.txt

4. Clone o SORT dentro da pasta do projeto

bashgit clone https://github.com/abewley/sort.git

5. Verifique a instalação

bashpython -c "
import torch, cv2, ultralytics, numpy
from sort.sort import Sort
print('torch:', torch.__version__, '| MPS:', torch.backends.mps.is_available())
print('opencv:', cv2.__version__)
print('ultralytics:', ultralytics.__version__)
print('SORT: ok')
"


Apple Silicon: MPS: True confirma que o sistema está usando a GPU do Mac, o que melhora significativamente a fluidez do processamento em tempo real.




▶️ Como rodar

bashconda activate visao_projeto
python app.py

Pressione Q para encerrar. O terminal exibe um resumo da sessão ao fechar.


🔧 Parâmetros de calibração

python# detector
CAMERA_INDEX      = 1       # 0 = webcam, 1 ou 2 = câmera externa
CONFIANCA_MINIMA  = 0.25    # limiar de confiança do YOLO
TEMPO_SAIDA       = 3.0     # segundos de ausência antes de encerrar cronômetro
PROCESSAR_A_CADA  = 2       # YOLO roda a cada N frames

# tracker
tracker = Sort(
    max_age=50,           # frames que o SORT mantém um ID sem nova detecção
    min_hits=1,           # detecções consecutivas para confirmar um ID novo
    iou_threshold=0.5     # sobreposição mínima para manter o mesmo ID
)

Guia de ajuste por problema

Sintoma observadoParâmetroDireçãoNão detecta pessoas longeCONFIANCA_MINIMA↓ diminuirMuitos falsos positivosCONFIANCA_MINIMA↑ aumentarID muda quando pessoa se movemax_age↑ aumentarIDs duplicados na mesma pessoaiou_threshold↑ aumentarSistema travando / lentoPROCESSAR_A_CADA↑ aumentar


🧩 Trechos principais do código

Detecção + Tracking

python# YOLO detecta apenas pessoas (classe 0)
resultados = model(frame,
                   classes=[0],
                   conf=CONFIANCA_MINIMA,
                   imgsz=480,
                   iou=0.45,
                   max_det=50)

# extrai coordenadas e confiança de cada detecção
deteccoes = []
for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        deteccoes.append([x1, y1, x2, y2, conf])

# SORT associa cada detecção a um ID único
tracks = tracker.update(np.array(deteccoes))

Lógica de permanência

python# registra timestamp de entrada do ID
if pid not in entrada:
    entrada[pid] = agora

# cronômetro individual em tempo real
tempo_atual = agora - entrada[pid]

# encerra ao detectar saída
if tempo_sem_ver > TEMPO_SAIDA:
    tempo_total = ultimo_visto[pid] - entrada[pid]
    historico[pid] = tempo_total

Filtro de falsos positivos

python# elimina objetos com proporção incompatível com humanos
proporcao = altura / (largura + 1e-5)
if largura < 30 or altura < 60:
    continue
if proporcao < 1.0 or proporcao > 3.5:
    continue


🔍 Desafios enfrentados e como foram resolvidos

Um dos aspectos mais relevantes do projeto foi o processo iterativo de calibração — cada problema exigiu análise do comportamento do sistema e ajuste direcionado de parâmetros.

DesafioCausaSolução aplicadaIDs duplicadosiou_threshold baixo — movimentos rápidos criavam ID novoAumentar de 0.3 para 0.6Perda de trackmax_age baixo — ausência curta encerrava o IDAumentar de 15 para 50 framesFalsos positivosObjetos verticais com silhueta parecida com pessoaFiltro de proporção da bounding boxPessoas sentadas/longeModelo nano insuficiente para poses variadasTrocar para yolov8x + ajustar iou=0.45 e max_det=50


⚠️ Limitações conhecidas


Câmera muito próxima compromete a detecção — o modelo perde a silhueta completa
Alta sobreposição entre pessoas pode causar troca de IDs
Parâmetros precisam ser recalibrados para cada ambiente
O sistema não identifica se a mesma pessoa retornou — cada aparição gera um novo ID



🚀 Próximos passos


 Dashboard web com Streamlit — interface com gráficos e tabelas interativas
 Persistência em banco de dados — histórico entre sessões
 Alertas automáticos — notificação por tempo de permanência
 Suporte a múltiplas câmeras — cobertura de ambientes maiores



📦 Dependências

torch==2.12.1
torchvision==0.27.1
ultralytics==8.4.47
opencv-python==4.13.0.92
numpy==2.2.6
filterpy==1.4.5
scikit-image==0.25.2
lap==0.5.13

## 👨‍💻 Autor

**Tiago Leite**
Disciplina de Visão Computacional — Pós-Graduação Data Science & IA - Faculdade SENAC