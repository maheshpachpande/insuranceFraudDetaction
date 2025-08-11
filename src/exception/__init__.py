# import sys
# import traceback

# def error_message_detail(error, error_detail=None):
#     """
#     Constructs a detailed error message with the filename and line number.

#     Args:
#         error (Exception): The original exception.
#         error_detail (traceback): Optional traceback object (from sys.exc_info()).

#     Returns:
#         str: Detailed error message.
#     """
#     if error_detail is None:
#         _, _, error_detail = sys.exc_info()

#     if error_detail is None:
#         return f"Error: {str(error)} (no traceback info available)"

#     file_name = error_detail.tb_frame.f_code.co_filename
#     line_number = error_detail.tb_lineno
#     error_message = (
#         f"Error occurred in Python script: [{file_name}] "
#         f"at line [{line_number}] with error message: [{str(error)}]"
#     )
#     return error_message

# class CustomException(Exception):
#     def __init__(self, error, error_detail=None):
#         """
#         Custom exception class for standardized error reporting.

#         Args:
#             error (Exception): Original exception instance.
#             error_detail (traceback, optional): Exception traceback.
#         """
#         super().__init__(str(error))
#         self.error_message = error_message_detail(error, error_detail)

#     def __str__(self):
#         return self.error_message

# # Test the CustomException
# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         raise CustomException(e)



# import sys

# def error_message_detail(error: Exception, error_detail: object) -> str: 
#     """
#     Returns error message with filename and line number
#     """
#     tb = error_detail  # This should already be the traceback object
#     file_name = tb.tb_frame.f_code.co_filename
#     line_number = tb.tb_lineno
#     return f"Error in file {file_name}, line {line_number}: {str(error)}"


# class CustomException(Exception):
#     def __init__(self, error_message: Exception, error_detail: object):
#         super().__init__(error_message)
#         self.error_message = error_message_detail(error_message, error_detail)

#     def __str__(self):
#         return self.error_message

# /home/mahesh/Desktop/insuranceFraudDetaction/src/exception/__init__.py
import sys

def error_message_detail(error, error_detail):
    tb = error_detail  # traceback object
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    return f"Error in file {file_name}, line {line_number}: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

