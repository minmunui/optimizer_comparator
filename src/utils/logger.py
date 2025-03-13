import datetime
import os
import sys
from enum import Enum, auto
from typing import Optional, TextIO


class LogLevel(Enum):
    """로그 레벨 정의"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def __str__(self) -> str:
        return self.name


class Logger:
    """
    글로벌 로깅 시스템

    이 클래스는 다양한 로그 레벨, 타임스탬프, 파일 로깅을 지원합니다.
    """

    _instance = None

    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """초기 설정"""
        self.enabled = True
        self.min_level = LogLevel.DEBUG
        self.log_file: Optional[TextIO] = None
        self.log_to_console = True
        self.include_timestamp = True

    def enable(self) -> None:
        """로깅 활성화"""
        self.enabled = True

    def disable(self) -> None:
        """로깅 비활성화"""
        self.enabled = False

    def set_level(self, level: LogLevel) -> None:
        """
        로그 레벨 설정

        Args:
            level: 출력할 최소 로그 레벨
        """
        self.min_level = level

    def start_file_logging(self, custom_filename: Optional[str] = None) -> None:
        """
        파일 로깅 시작

        Args:
            custom_filename: 사용자 지정 파일명. 없으면 타임스탬프 형식 사용
        """
        if self.log_file:
            self.log_file.close()

        if custom_filename:
            filename = custom_filename
        else:
            # yyyy_MM_dd_hh_mm_ss.txt 형식의 파일명 생성
            now = datetime.datetime.now()
            filename = now.strftime("%Y_%m_%d_%H_%M_%S.txt")

        # 로그 디렉토리 확인 및 생성
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        filepath = os.path.join(log_dir, filename)
        self.log_file = open(filepath, 'a', encoding='utf-8')
        self.log(LogLevel.INFO, f"📄: Log is recorded in '{filepath}'")

    def stop_file_logging(self) -> None:
        """파일 로깅 중지"""
        if self.log_file:
            self.log(LogLevel.INFO, "파일 로깅을 중지합니다.")
            self.log_file.close()
            self.log_file = None

    def set_console_output(self, enabled: bool) -> None:
        """콘솔 출력 설정"""
        self.log_to_console = enabled

    def set_timestamp(self, enabled: bool) -> None:
        """타임스탬프 포함 여부 설정"""
        self.include_timestamp = enabled

    def log(self, level: LogLevel, message: str) -> None:
        """
        지정된 레벨로 메시지 로깅

        Args:
            level: 로그 레벨
            message: 로그 메시지
        """
        if not self.enabled or level.value < self.min_level.value:
            return

        # 타임스탬프 추가
        timestamp = ""
        if self.include_timestamp:
            now = datetime.datetime.now()
            timestamp = f"[{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "

        # 로그 형식: [시간] [레벨] 메시지
        formatted_message = f"{timestamp}[{level}] {message}"

        # 콘솔에 출력
        if self.log_to_console:
            if level in (LogLevel.ERROR, LogLevel.CRITICAL):
                print(formatted_message, file=sys.stderr)
            else:
                print(formatted_message)

        # 파일에 기록
        if self.log_file:
            self.log_file.write(formatted_message + "\n")
            self.log_file.flush()

    def debug(self, message: str) -> None:
        """디버그 레벨 로그"""
        self.log(LogLevel.DEBUG, message)

    def info(self, message: str) -> None:
        """정보 레벨 로그"""
        self.log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        """경고 레벨 로그"""
        self.log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        """에러 레벨 로그"""
        self.log(LogLevel.ERROR, message)

    def critical(self, message: str) -> None:
        """치명적 에러 레벨 로그"""
        self.log(LogLevel.CRITICAL, message)


# 전역 로거 인스턴스 생성
def get_logger() -> Logger:
    """전역 로거 인스턴스 반환"""
    return Logger()


# 편의성을 위한 함수들
def debug(message: str) -> None:
    """디버그 레벨 로그"""
    get_logger().debug(message)


def info(message: str) -> None:
    """정보 레벨 로그"""
    get_logger().info(message)


def warning(message: str) -> None:
    """경고 레벨 로그"""
    get_logger().warning(message)


def error(message: str) -> None:
    """에러 레벨 로그"""
    get_logger().error(message)


def critical(message: str) -> None:
    """치명적 에러 레벨 로그"""
    get_logger().critical(message)