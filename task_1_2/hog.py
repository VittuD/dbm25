import cv2
import numpy as np
import torch

def hog(img: torch.Tensor):
  img= cv2.resize(img,(300,10))

  #Gradienti + maschere della traccia
  gx=cv2.filter2D(img.astype(np.float32), -1, np.array([[-1,0,1]]))
  gy= cv2.filter2D(img.astype(np.float32), -1, np.array([[-1],[0],[1]])) #T di [-1,0,1]


  #Magnitudine 
  magnitude = np.sqrt(gx**2 + gy**2)
  angle = np.arctan2(gy, gx) * (180 / np.pi)  # Radiani → gradi
  angle[angle < 0] += 360  # Range 0-360


  # Parametri della griglia
  cell_rows, cell_cols = 10, 10
  cell_height = img.shape[0] // cell_rows   # 10
  cell_width  = img.shape[1] // cell_cols   # 30

  bins = 9
  bin_size = 360 // bins  # 40 gradi per bin

  hog_features = []

  #Griglia
  for i in range(cell_rows):
    for j in range(cell_cols):
      y_start, y_end = i * cell_height, (i + 1) * cell_height
      x_start, x_end = j * cell_width,  (j + 1) * cell_width

      # Estraggo sottoimmagine (cella)
      mag_cell = magnitude[y_start:y_end, x_start:x_end]
      ang_cell = angle[y_start:y_end, x_start:x_end]

      # Istogramma a 9 bin (da 0 a 360°)
      hist = np.zeros(bins)

      # Popola istogramma 
      for y in range(mag_cell.shape[0]):
        for x in range(mag_cell.shape[1]):
          mag = mag_cell[y, x]
          ang = ang_cell[y, x]
          bin_idx = int(ang // bin_size) % bins
          hist[bin_idx] += mag

      hog_features.extend(hist)

  return torch.tensor(hog_features)