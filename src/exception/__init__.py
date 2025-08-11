
import sys

# def error_message_detail(error, error_detail):
#     tb = error_detail  # traceback object
#     file_name = tb.tb_frame.f_code.co_filename
#     line_number = tb.tb_lineno
#     return f"Error in file {file_name}, line {line_number}: {str(error)}"


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = sys.exc_info()  # exc_tb is the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error in script: [{file_name}] line [{exc_tb.tb_lineno}] message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

