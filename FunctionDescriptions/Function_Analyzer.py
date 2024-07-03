import os
import understand
from database_functions import capture_files, capture_functions, sort_functions, UND_PATH
from llm_withContext import describe_functions
# from llm_noContext import describe_functions

if __name__ == '__main__':
    # Open Database
    try:
        db = understand.open(UND_PATH)
        print("Sucessfully opened Database")
    except:
        print("Couln't open DB")
    # Capture data
    captured_files = capture_files(db)
    captured_functions = capture_functions(db)
    #sort data
    sorted = sort_functions(captured_files, captured_functions)

    describe_functions(sorted)

    #close db
    db.close()