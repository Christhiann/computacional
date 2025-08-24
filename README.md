# Detecção de Manipulações em Imagens (CoMoFoD)

Este projeto utiliza **técnicas básicas de segmentação** para detectar regiões potencialmente adulteradas em imagens forjadas, gerando **máscaras explicáveis** e métricas de comparação.

---

## Método

- Carrega imagens **original (O), manipulada (F) e máscara real (B)**  
- Converte a imagem forjada para **HSV**  
- Aplica **K-means (K=3)** para segmentar regiões de interesse  
- Gera máscara binária das áreas detectadas  
- Opcional: **refinamento morfológico** para reduzir ruídos  
- Calcula métricas: **Acurácia, IoU, Precisão e Recall**  
- Visualiza **original, forjada, máscara real e máscara gerada lado a lado**

---

## Como usar

```bash
pip install opencv-python numpy matplotlib

