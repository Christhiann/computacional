import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Arquivos ===
orig_path = "027_O.png"   # original
forg_path = "027_F.png"   # imagem manipulada
mask_real_path = "027_B.png"  # máscara real

# === Carregar imagens ===
orig = cv2.imread(orig_path)
forg = cv2.imread(forg_path)
mask_real = cv2.imread(mask_real_path, cv2.IMREAD_GRAYSCALE)

# Converter para RGB para exibição
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
forg_rgb = cv2.cvtColor(forg, cv2.COLOR_BGR2RGB)

# === Converter para HSV para aplicar K-means ===
img_hsv = cv2.cvtColor(forg, cv2.COLOR_BGR2HSV)
Z = img_hsv.reshape((-1,3))
Z = np.float32(Z)

# === Aplicar K-means ===
K = 3   # número de clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

labels = labels.flatten()
mask_pred = (labels.reshape(forg.shape[:2]) == 1).astype(np.uint8) * 255   # pode trocar ==1 por ==0 ou ==2

# === Normalizar para 0 e 1 ===
mask_real_bin = (mask_real > 127).astype(np.uint8)
mask_pred_bin = (mask_pred > 127).astype(np.uint8)

# === Calcular métricas ===
intersection = np.logical_and(mask_real_bin, mask_pred_bin).sum()
union = np.logical_or(mask_real_bin, mask_pred_bin).sum()
iou = intersection / union if union != 0 else 0
accuracy = (mask_real_bin == mask_pred_bin).sum() / mask_real_bin.size

print(f"Acurácia: {accuracy:.3f}")
print(f"IoU: {iou:.3f}")

# === Mostrar resultados lado a lado ===
titles = ["Original (O)", "Forjada (F)", "Máscara Real (B)", "Máscara Gerada (K-means)"]
images = [orig_rgb, forg_rgb, mask_real, mask_pred]

plt.figure(figsize=(14,6))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i], cmap="gray" if i>=2 else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
# === Gerar Relatório em TXT ===
with open("relatorio_analise.txt", "w", encoding="utf-8") as f:
    f.write("RELATÓRIO DE ANÁLISE DE INTEGRIDADE DE IMAGEM\n")
    f.write("="*50 + "\n\n")
    f.write(f"Arquivo original: {orig_path}\n")
    f.write(f"Arquivo forjado: {forg_path}\n")
    f.write(f"Arquivo máscara real: {mask_real_path}\n\n")
    f.write(f"Acurácia: {accuracy:.3f}\n")
    f.write(f"IoU: {iou:.3f}\n\n")
    f.write("Observações:\n")
    f.write("- A máscara gerada foi obtida via segmentação K-means.\n")
    f.write("- A acurácia pode estar inflada pelo grande número de pixels de fundo.\n")
    f.write("- A IoU mede a sobreposição real entre adulteração prevista e real.\n\n")

print("✅ Relatório salvo em relatorio_analise.txt e imagem em comparacao_resultados.png")

