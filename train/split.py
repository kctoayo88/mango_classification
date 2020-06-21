import csv
import shutil

with open('./train/train.csv', newline = '') as csvfile:
  # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
  rows = csv.DictReader(csvfile)

  # 以迴圈輸出指定欄位
  for row in rows:
    if row['label'] == 'A':
        shutil.copy('./train/train/' + row['image_id'], './train/data/1/')
    elif row['label'] == 'B':
        shutil.copy('./train/train/' + row['image_id'], './train/data/2/')
    elif row['label'] == 'C':
        shutil.copy('./train/train/' + row['image_id'], './train/data/3/')

with open('./train/valid.csv', newline = '') as csvfile:
  # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
  rows = csv.DictReader(csvfile)

  # 以迴圈輸出指定欄位
  for row in rows:
    if row['label'] == 'A':
        shutil.copy('./train/valid/' + row['image_id'], './train/data/1/')
    elif row['label'] == 'B':
        shutil.copy('./train/valid/' + row['image_id'], './train/data/2/')
    elif row['label'] == 'C':
        shutil.copy('./train/valid/' + row['image_id'], './train/data/3/')