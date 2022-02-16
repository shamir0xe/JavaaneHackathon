from src.delegators.amlak_delegator import AmlakDelegator

def main():
    AmlakDelegator() \
    .read_data() \
    .trim_data() \
    .select_columns() \
    .normalize_data() \
    .backup_data() \
    .append_features() \
    .append_numerical_data() \
    .ml_procedure() \
    .predict_result() \
    .read_data() \
    .output()


if __name__ == '__main__':
    main()