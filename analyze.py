import pandas as pd
import numpy as np


def find_best_params(csv_file_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)

    # 각 mutation_rate와 crossover_rate 조합별로 value의 중앙값 계산
    grouped = df.groupby(['mutation_rate', 'crossover_rate'])
    median_values = grouped['value'].median().reset_index()

    # 중앙값이 가장 높은 조합 찾기
    best_combination = median_values.loc[median_values['value'].idxmax()]

    # 결과 출력
    print(f"분석된 총 레코드 수: {len(df)}")
    print(f"고유한 mutation_rate 개수: {df['mutation_rate'].nunique()}")
    print(f"고유한 crossover_rate 개수: {df['crossover_rate'].nunique()}")
    print(f"고유한 조합 개수: {len(grouped)}")
    print("\n최적의 파라미터 조합:")
    print(f"mutation_rate: {best_combination['mutation_rate']}")
    print(f"crossover_rate: {best_combination['crossover_rate']}")
    print(f"value 중앙값: {best_combination['value']}")

    # 상위 5개 조합 출력
    print("\n상위 5개 조합:")
    top5 = median_values.sort_values('value', ascending=False).head(5)
    for idx, row in top5.iterrows():
        print(
            f"mutation_rate: {row['mutation_rate']}, crossover_rate: {row['crossover_rate']}, value 중앙값: {row['value']}")

    # 추가 분석: 각 파라미터별 분포 확인
    print("\n각 파라미터별 최적값 분포:")

    # mutation_rate별 최적 성능
    mut_analysis = df.groupby('mutation_rate')['value'].median().sort_values(ascending=False)
    print("\nmutation_rate별 최적 성능(중앙값):")
    print(mut_analysis.head())

    # crossover_rate별 최적 성능
    cross_analysis = df.groupby('crossover_rate')['value'].median().sort_values(ascending=False)
    print("\ncrossover_rate별 최적 성능(중앙값):")
    print(cross_analysis.head())

    return best_combination


if __name__ == "__main__":
    # CSV 파일의 경로를 지정하세요
    csv_file_path = "summary.csv"  # 실제 파일 경로로 변경하세요
    best_params = find_best_params(csv_file_path)