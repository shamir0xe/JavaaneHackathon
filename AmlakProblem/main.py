from src.delegators.amlak_delegator import AmlakDelegator

def main():
    AmlakDelegator() \
    .read_data() \
    .trim_data() \
    .select_columns() \
    .append_features() \
    .normalize_data() \
    .ml_procedure() \
    .output()


if __name__ == '__main__':
    main()