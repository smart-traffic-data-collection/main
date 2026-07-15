import torch

model_path = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/phase4_unfrozen_tuning-2/weights/best.pt"
print(f"Loading weights from: {model_path}")

ckpt = torch.load(model_path, weights_only=False)

# The final, true mapping based on your visual confirmation
ckpt['model'].names = {0: 'bicycle', 1: 'car', 2: 'person'}

torch.save(ckpt, model_path)

print(f"Dictionary perfectly aligned! Final internal names: {ckpt['model'].names}")
