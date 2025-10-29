import pandas as pd
from config import cfg

# config.py에 설정된 학습용 어노테이션 파일 경로를 가져옵니다.
csv_path = cfg.train['annotation_file']

print(f"--- Checking CSV Header for: {csv_path} ---")

try:
    # 데이터 로더가 파일을 읽는 방식과 최대한 유사하게 읽어봅니다.
    # 원본 파일일 수 있으므로 skiprows=9 옵션을 사용합니다.
    df = pd.read_csv(csv_path, skiprows=9)
    
    print("\n[Found Columns]")
    print(df.columns.tolist())
    print("\n[Data Preview (First 2 Rows)]")
    print(df.head(2).to_string())

except FileNotFoundError:
    print(f"Error: File not found at '{csv_path}'. Please check the path in config.py.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")

print("\n--- Check Complete ---")