from src.delegators.text_delegator import TextDelegator
from src.delegators.amlak_delegator import AmlakDelegator
from src.delegators.picture_delegator import PictureDelegator
from src.facades.terminal_arguments import TerminalArguments

def main():
    if TerminalArguments.exists('text'):
        # TEXT SECTION
        TextDelegator() \
        .read_data() \
        .backup_data() \
        .set_config() \
        .select_columns() \
        .generate_database() \
        .ml_procedure() \
        .predict() \
        .join_dataset() \
        .drop_redundants() \
        .restore_data() \
        .split_dataset() \
        .output()
    elif TerminalArguments.exists('picture'):
        # PICTURE SECTION
        PictureDelegator() \
        .set_config() \
        .build_graph()
        # .dimension_reduction() \
        # .test()
    else:
        # AMLAK SECTION
        AmlakDelegator() \
        .read_data() \
        .select_columns() \
        .append_pic_graph() \
        .append_diff_time() \
        .normalize_data() \
        .backup_data() \
        .append_features() \
        .append_bc_data() \
        .edit_columns() \
        .modify_labels() \
        .normalize_data() \
        .ml_procedure() \
        .predict_result() \
        .read_data() \
        .output()

if __name__ == '__main__':
    main()
