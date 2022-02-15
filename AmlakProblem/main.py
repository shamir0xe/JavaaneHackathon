from src.delegators.amlak_delegator import AmlakDelegator

def main():
    AmlakDelegator() \
    .read_data() \
    .trim_data() \
    .select_columns() \
    .normalize_data() \
    .bare_tokens() \
    .append_features() \
    .append_numerical_data() \
    .ml_procedure() \
    .predict_result() \
    .read_data() \
    .output()
    # .dimension_reduction() \


if __name__ == '__main__':
    main()