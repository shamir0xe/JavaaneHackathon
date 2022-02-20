import sys


class TerminalArguments:
    @staticmethod
    def exists(option: str) -> bool:
        return f'--{option}' in sys.argv[1:]
    
