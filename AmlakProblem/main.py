from src.delegators.amlak_delegator import AmlakDelegator

def main():
    AmlakDelegator() \
    .read_data() \
    .trim_data() \
    .select_columns() \
    .normalize_data() \
    .ml_procedure() \
    .output()
    # .append_features() \

if __name__ == '__main__':
    main()