import sys


def error_message_detail(error,error_detail:sys, additional_info = None):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    if additional_info is not None:
        error_message=f"\n**************\n\nError in script file:\n'{file_name}'.\nLine number:\n'{exc_tb.tb_lineno}'.\nWith error message:\n---->>>> '{str(error)}' <<<<----\nAditional info: \n---->>>> '{additional_info}' <<<<----\n\n**************"
    
    else:
        error_message=f"\n**************\nError in script file:\n'{file_name}'.\nLine number:\n'{exc_tb.tb_lineno}'.\nWith error message:\n---->>>> '{str(error)}' <<<<----\n**************"
    #     error_message="\n**************\nError in script file:\n'{0}'.\nLine number:\n'{1}'.\nWith error message:\n---->>>> '{2}' <<<<----\n**************".format(
    #  file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys, additional_info = None):
        super().__init__(error_message)
        if additional_info is not None:
            self.error_message=error_message_detail(error_message,error_detail=error_detail,additional_info=additional_info)            
        else:
            self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message