@echo off
setlocal enabledelayedexpansion

:: 파라미터 값 설정
set mutation_rate_values=0.01 0.03 0.05 0.07 0.10 0.15 0.20
set crossover_rate=1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
set seeds=1 2 3 4 5 6 7 8 9 10


:: 출력 결과를 저장할 파일 지정
set output_file=results.txt

:: 결과 파일 초기화
echo Results for all parameter combinations > %output_file%
echo ---------------------------------------- >> %output_file%

:: 모든 조합에 대해 반복 실행
for %%a in (%mutation_rate_values%) do (
    for %%b in (%crossover_rate%) do (
        for %%c in (%seeds%) do (
            echo Running: python main.py --solver ga --ga_mutation_rate %%a --ga_crossover_rate %%b --random_seed %%c --fraction 4

            :: 명령 실행 및 결과 저장
            echo Command: python main.py --solver ga --ga_mutation_rate %%a --ga_crossover_rate %%b --random_seed %%c --fraction 4 >> %output_file%


            call .venv\Scripts\activate.bat && (
                python main.py --solver ga --ga_mutation_rate %%a --ga_crossover_rate %%b --random_seed %%c --fraction 4 >> %output_file% 2>&1
                :: 가상환경 비활성화
                call deactivate
            )

            echo. >> %output_file%
            echo. >> %output_file%
        )
    )
)

echo All combinations completed! Results saved to %output_file%