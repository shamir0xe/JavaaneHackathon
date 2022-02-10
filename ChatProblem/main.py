from src.delegators.chat_delegator import ChatDelegator

def main():
    ChatDelegator() \
    .read_database() \
    .test()

if __name__ == '__main__':
    main()
