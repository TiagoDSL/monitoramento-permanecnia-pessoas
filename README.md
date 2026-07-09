# 👁️ Sistema de Monitoramento de Permanência de Pessoas em Tempo Real

Sistema de visão computacional que detecta, rastreia e cronometra o tempo de permanência de pessoas em ambientes fechados — em tempo real, com apenas uma câmera comum.


Projeto desenvolvido como parte do portfólio em Ciência de Dados e Visão Computacional.




## 📸 Sistema em funcionamento

<img width="1276" height="743" alt="Captura de Tela 2026-07-02 às 15 41 43" src="https://github.com/user-attachments/assets/09dca8f3-9556-416b-b073-5c317f1eec94" />


> **Sistema em execução:** detecção simultânea de múltiplas pessoas, rastreamento por IDs únicos, cronômetros individuais e painel de métricas atualizado em tempo real.

---

# 💡 Problema

Estabelecimentos como lojas, restaurantes, clínicas e museus geralmente não possuem informações sobre:

- Quantas pessoas estão presentes no ambiente;
- Quanto tempo cada pessoa permanece;
- Como a ocupação varia ao longo do dia.

A coleta manual dessas informações é inviável, enquanto soluções comerciais costumam exigir hardware dedicado e alto investimento.

Este projeto demonstra como obter essas métricas utilizando apenas uma câmera comum e técnicas de Visão Computacional.

---

# ✅ Solução

O sistema executa automaticamente todo o fluxo de monitoramento:

- 👤 Detecta pessoas em tempo real utilizando **YOLOv8**;
- 🆔 Atribui um identificador único para cada indivíduo com **SORT**;
- ⏱️ Calcula o tempo de permanência de cada pessoa;
- 📊 Exibe métricas consolidadas durante toda a execução.

---

# 🌍 Possíveis aplicações

| Setor | Aplicação |
|--------|-----------|
| 🛍️ Varejo | Tempo médio de permanência por seção da loja |
| 🍽️ Restaurantes | Rotatividade de mesas |
| 🏥 Clínicas e hospitais | Monitoramento de salas de espera |
| 🖼️ Museus | Análise de interesse por exposições |
| 🔒 Segurança | Permanência em áreas restritas |
| 🎓 Universidades | Monitoramento de fluxo em laboratórios e bibliotecas |

---

# 🧠 Competências demonstradas

Além da implementação do algoritmo, o projeto evidencia conhecimentos em:

- Visão Computacional;
- Rastreamento de objetos (*Object Tracking*);
- Processamento de vídeo em tempo real;
- Integração entre modelos de IA;
- Engenharia de Software aplicada à IA;
- Calibração de modelos baseada em experimentação;
- Tratamento de falsos positivos;
- Geração de métricas analíticas.

---

# 🛠️ Stack tecnológica

| Tecnologia | Função |
|------------|--------|
| Python 3.10 | Linguagem principal |
| YOLOv8x | Detecção de pessoas |
| SORT | Rastreamento por IDs |
| OpenCV | Captura e renderização dos frames |
| NumPy | Manipulação de matrizes |
| PyTorch | Backend para inferência do YOLO |

---

# ⚙️ Principais decisões técnicas

## YOLOv8x em vez do YOLOv8n

Durante os testes, o modelo **YOLOv8 Nano** apresentou dificuldades para detectar:

- pessoas distantes;
- pessoas parcialmente ocultas;
- pessoas sentadas.

Apesar do maior custo computacional, o **YOLOv8 Extra Large (YOLOv8x)** apresentou desempenho significativamente superior, tornando-se a melhor escolha para este cenário.

O impacto computacional foi compensado utilizando aceleração via **Apple Metal Performance Shaders (MPS)**.

---

## Utilização do SORT

O YOLO detecta pessoas individualmente em cada frame, porém não mantém identidade entre eles.

O **SORT** resolve esse problema ao:

- prever a posição dos objetos utilizando **Filtro de Kalman**;
- associar novas detecções aos IDs existentes;
- manter o mesmo identificador durante toda a permanência da pessoa.

Essa etapa torna possível calcular o tempo individual de permanência.

---

# 🏗️ Arquitetura do pipeline

```text
📷 Câmera
      │
      ▼
🔍 YOLOv8x
(detecção de pessoas)
      │
      ▼
🔄 SORT
(rastreamento por IDs)
      │
      ▼
⏱️ Controle de permanência
(registro de entrada e saída)
      │
      ▼
📊 Dashboard OpenCV
(métricas em tempo real)
```

---

# 📈 Métricas geradas

Durante a execução, o sistema disponibiliza:

- 👥 Pessoas atualmente no ambiente;
- 📌 Total de pessoas detectadas na sessão;
- ⏱️ Tempo médio de permanência;
- ⬇️ Menor permanência registrada;
- ⬆️ Maior permanência registrada;
- 🆔 Lista de IDs ativos com cronômetro individual.

---

# ⚙️ Instalação

## Pré-requisitos

- Python 3.10
- Git
- Miniforge (Apple Silicon)

---

## 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/monitoramento-permanencia.git

cd monitoramento-permanencia
```

---

## 2. Crie o ambiente virtual

```bash
conda create -n visao_projeto python=3.10 -y

conda activate visao_projeto
```

---

## 3. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

---

## 4. Clone o SORT

```bash
git clone https://github.com/abewley/sort.git
```

---

## 5. Verifique a instalação

```bash
python -c "
import torch
import cv2
import ultralytics
import numpy
from sort.sort import Sort

print('torch:', torch.__version__, '| MPS:', torch.backends.mps.is_available())
print('opencv:', cv2.__version__)
print('ultralytics:', ultralytics.__version__)
print('SORT: ok')
"
```

> **Apple Silicon:** se `MPS: True` for exibido, a aplicação utilizará a GPU integrada para acelerar a inferência.

---

# ▶️ Execução

```bash
conda activate visao_projeto

python app.py
```

Durante a execução:

- a janela exibirá o vídeo em tempo real;
- o painel lateral apresentará as métricas atualizadas;
- pressione **Q** para encerrar a aplicação.

Ao finalizar, um resumo da sessão será exibido no terminal.

---

# 🎛️ Configuração

Os principais parâmetros do detector e do tracker podem ser ajustados para diferentes ambientes.

### Detector (YOLO)

```python
CAMERA_INDEX = 1
CONFIANCA_MINIMA = 0.25
TEMPO_SAIDA = 3.0
PROCESSAR_A_CADA = 2
```

| Parâmetro | Descrição |
|-----------|-----------|
| `CAMERA_INDEX` | Seleciona a câmera utilizada. |
| `CONFIANCA_MINIMA` | Confiança mínima da detecção. |
| `TEMPO_SAIDA` | Tempo sem detecção antes de considerar a saída da pessoa. |
| `PROCESSAR_A_CADA` | Executa o detector a cada N frames. |

---

### Tracker (SORT)

```python
tracker = Sort(
    max_age=50,
    min_hits=1,
    iou_threshold=0.5
)
```

| Parâmetro | Descrição |
|-----------|-----------|
| `max_age` | Tempo máximo para manter um ID sem novas detecções. |
| `min_hits` | Número mínimo de detecções para confirmar um novo ID. |
| `iou_threshold` | IoU mínimo para associar uma nova detecção ao mesmo ID. |

---

## 🛠️ Guia rápido de calibração

| Situação | Ajuste recomendado |
|-----------|-------------------|
| Pessoas distantes não são detectadas | Reduzir `CONFIANCA_MINIMA` |
| Muitos falsos positivos | Aumentar `CONFIANCA_MINIMA` |
| IDs mudam durante movimentos | Aumentar `max_age` |
| IDs duplicados | Aumentar `iou_threshold` |
| Processamento lento | Aumentar `PROCESSAR_A_CADA` |

# 🧩 Principais componentes da implementação

## 🎯 Detecção e rastreamento de pessoas

A detecção é realizada pelo **YOLOv8**, configurado para identificar apenas pessoas (`classe 0`). Em seguida, as detecções são enviadas ao **SORT**, responsável por manter um identificador único (ID) para cada indivíduo durante sua permanência na cena.

```python
resultados = model(
    frame,
    classes=[0],
    conf=CONFIANCA_MINIMA,
    imgsz=480,
    iou=0.45,
    max_det=50
)

deteccoes = []
for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        deteccoes.append([x1, y1, x2, y2, conf])

tracks = tracker.update(np.array(deteccoes))
```

**O que acontece nesta etapa?**

- O YOLO detecta apenas pessoas.
- Cada detecção gera uma *bounding box* e um nível de confiança.
- O SORT associa cada detecção a um ID persistente, permitindo acompanhar cada pessoa entre os frames.

---

## ⏱️ Controle do tempo de permanência

Cada pessoa recebe um timestamp no momento em que é detectada pela primeira vez. Enquanto permanecer visível, seu tempo é atualizado continuamente.

```python
if pid not in entrada:
    entrada[pid] = agora

tempo_atual = agora - entrada[pid]

if tempo_sem_ver > TEMPO_SAIDA:
    tempo_total = ultimo_visto[pid] - entrada[pid]
    historico[pid] = tempo_total
```

**Fluxo da lógica**

1. Registra o instante de entrada.
2. Atualiza o cronômetro em tempo real.
3. Detecta a saída da pessoa.
4. Salva o tempo total de permanência.

---

## 🛡️ Filtro de falsos positivos

Para reduzir detecções incorretas, o sistema aplica filtros geométricos baseados nas dimensões da *bounding box*.

```python
proporcao = altura / (largura + 1e-5)

if largura < 30 or altura < 60:
    continue

if proporcao < 1.0 or proporcao > 3.5:
    continue
```

Esses filtros ajudam a eliminar:

- objetos estreitos;
- postes e placas;
- sombras;
- estruturas verticais confundidas com pessoas.

---

# 🔍 Desafios encontrados

Durante o desenvolvimento, diversos testes foram realizados para calibrar o detector e o rastreador, buscando o melhor equilíbrio entre precisão e estabilidade.

| Desafio | Causa | Solução adotada |
|---------|--------|-----------------|
| IDs duplicados | `iou_threshold` muito baixo fazia o tracker criar novos IDs durante movimentos rápidos | Aumento do `iou_threshold` |
| Perda de rastreamento | `max_age` pequeno removia IDs rapidamente | Aumento para 50 frames |
| Falsos positivos | Objetos verticais semelhantes a pessoas | Inclusão de filtros por tamanho e proporção |
| Pessoas distantes ou sentadas | Modelo menor apresentava menor capacidade de generalização | Utilização do **YOLOv8x** e ajuste dos parâmetros de detecção |

---

# ⚠️ Limitações

Apesar dos bons resultados, o sistema ainda apresenta algumas limitações conhecidas:

- A câmera deve capturar o corpo inteiro da pessoa para maximizar a precisão.
- Grande sobreposição entre pessoas pode ocasionar troca de IDs.
- Os parâmetros precisam ser recalibrados para diferentes ambientes e posicionamentos de câmera.
- O sistema não realiza reidentificação (*Re-Identification*). Assim, uma pessoa que sai do campo de visão e retorna recebe um novo ID.

---

# 🚀 Melhorias e implementos interessantes

- 📊 Dashboard em Streamlit para visualização das métricas em tempo real.
- 💾 Persistência dos dados em banco de dados.
- 🔔 Alertas automáticos para permanência acima de um limite configurável.
- 📷 Suporte a múltiplas câmeras.
- 📈 Estatísticas históricas de ocupação e tempo médio de permanência.
- 🧠 Integração com técnicas de Person Re-Identification (ReID).

---

# 📦 Dependências

| Biblioteca | Versão |
|------------|---------|
| torch | 2.12.1 |
| torchvision | 0.27.1 |
| ultralytics | 8.4.47 |
| opencv-python | 4.13.0.92 |
| numpy | 2.2.6 |
| filterpy | 1.4.5 |
| scikit-image | 0.25.2 |
| lap | 0.5.13 |

## 👨‍💻 Autor

**Tiago Leite**
Disciplina de Visão Computacional — Pós-Graduação Data Science & IA - Faculdade SENAC
