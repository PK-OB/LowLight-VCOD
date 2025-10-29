import pandas as pd
import json
from tqdm import tqdm

# 입력 파일과 출력 파일 이름
input_csv_path = 'data/MoCA/Annotations/annotations_modified.csv'
output_csv_path = 'data/MoCA/Annotations/annotations_cleaned.csv'

print(f"Reading '{input_csv_path}'...")
try:
    df = pd.read_csv(input_csv_path, comment='#')
except Exception:
    df = pd.read_csv(input_csv_path)

# 파일 이름 열을 자동으로 찾기
if '#filename' in df.columns:
    filename_col = '#filename'
elif 'file_list' in df.columns:
    filename_col = 'file_list'
elif 'filename' in df.columns:
    filename_col = 'filename'
else:
    raise ValueError("Annotation file must contain a filename column.")

print("Processing rows...")
cleaned_data = []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    if 'spatial_coordinates' in row and isinstance(row['spatial_coordinates'], str) and row['spatial_coordinates'] != '[]':
        try:
            coords_str = row['spatial_coordinates'].strip('[]')
            
            # ▼▼▼ 수정된 부분 ▼▼▼
            # 먼저 float으로 변환한 뒤, 각 숫자를 int로 다시 변환합니다.
            coords = [int(float(c)) for c in coords_str.split(',')]
            
            shape_id = coords[0]
            if shape_id == 2 and len(coords) >= 5:
                x, y, w, h = coords[1], coords[2], coords[3], coords[4]
                
                all_points_x = [x, x + w, x + w, x]
                all_points_y = [y, y, y + h, y + h]
                
                cleaned_data.append({
                    'filename': row[filename_col],
                    'all_points_x': json.dumps(all_points_x),
                    'all_points_y': json.dumps(all_points_y)
                })
        except (ValueError, IndexError):
            continue

# 새로운 데이터프레임 생성 및 파일로 저장
cleaned_df = pd.DataFrame(cleaned_data)
if cleaned_df.empty:
    print("\nWarning: No valid annotations found. The output file will be empty.")
else:
    cleaned_df.to_csv(output_csv_path, index=False)

print(f"\nProcessing complete!")
print(f"Total {len(df)} rows processed.")
print(f"{len(cleaned_df)} valid annotation rows saved to '{output_csv_path}'.")