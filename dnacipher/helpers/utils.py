""" General utilities used across the dnacipher package.
"""

import time

def finalize_log(log_file, script_start_time):
    """ Finalizes the logging.
    """

    script_end_time = time.time()

    total_minutes = round((script_end_time-script_start_time)/60, 3)
    total_hours = round((script_end_time-script_start_time)/60/60, 3)

    print("DONE.", file=log_file, flush=True)
    print("TOTAL minutes: ", total_minutes, file=log_file, flush=True )
    print("TOTAL hours: ", total_hours, file=log_file, flush=True )

    log_file.close()



