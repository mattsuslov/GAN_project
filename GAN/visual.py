import matplotlib.pyplot as plt
import numpy as np

def show_batch(pred):
  fig=plt.figure(figsize=(10, 10))
  cnt = 0
  for img in pred:
    if cnt > 64:
      break
    img = np.transpose(img.detach().cpu(), (1, 2, 0))
    cnt += 1
    fig.add_subplot(8, 8, cnt)
    plt.axis('off')
    plt.imshow(img)

  plt.show()